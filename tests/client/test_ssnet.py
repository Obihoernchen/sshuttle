import queue
import random
import select
import socket
import struct
import time
from unittest.mock import Mock

import pytest

import sshuttle.ssnet as ssnet


def _drain(timeout=5.0):
    """Wait for the close helpers to work through the queue."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ssnet._close_q is None or ssnet._close_q.empty():
            return True
        time.sleep(0.01)
    return False


def test_close_later_closes_the_socket():
    a, b = socket.socketpair()
    try:
        ssnet.close_later(a)
        assert _drain()
        # give the helper a moment to actually run close() after the get()
        deadline = time.monotonic() + 5.0
        while a.fileno() != -1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert a.fileno() == -1
    finally:
        b.close()


def test_close_later_tolerates_none_and_double_close():
    ssnet.close_later(None)
    s, other = socket.socketpair()
    s.close()
    ssnet.close_later(s)  # already closed; must not raise
    assert _drain()
    other.close()


def test_close_later_closes_inline_when_the_backlog_is_full():
    """A full queue means the closers cannot keep up; holding the fd open
    would be worse than taking the stall here."""
    class FullQueue:
        # raises from put() too, so a regression fails instead of hanging
        def put_nowait(self, item):
            raise queue.Full

        put = put_nowait

    a, b = socket.socketpair()
    saved = ssnet._close_q
    ssnet._close_q = FullQueue()
    try:
        ssnet.close_later(a)
        assert a.fileno() == -1  # closed inline, not queued
    finally:
        ssnet._close_q = saved
        b.close()


def test_handler_dispose_is_a_noop():
    ssnet.Handler(socks=[]).dispose()


def test_proxy_dispose_leaves_the_mux_alone():
    """A MuxWrapper's rsock/wsock are the mux's own stdin/stdout, and must
    never be handed to close_later()."""
    closed = []

    class FakeSock:
        def __init__(self, name):
            self.name = name

        def fileno(self):
            return -1

        def close(self):
            closed.append(self.name)

    def wrap(cls, name):
        # __new__ skips __init__, but these wrappers still run __del__ (and
        # through it __repr__ and MuxWrapper.nowrite()) when the collector
        # gets to them, so fill in the attributes those touch.
        w = cls.__new__(cls)
        w.rsock = w.wsock = FakeSock(name)
        w.exc = None
        w.peername = name
        w.shut_read = w.shut_write = True
        w.channel = 0  # MuxWrapper.__repr__ wants one
        return w

    proxy = ssnet.Proxy.__new__(ssnet.Proxy)
    proxy.wrap1 = wrap(ssnet.MuxWrapper, 'mux')
    proxy.wrap2 = wrap(ssnet.SockWrapper, 'real')
    proxy.dispose()
    assert _drain()

    deadline = time.monotonic() + 5.0
    while 'real' not in closed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert closed == ['real']


@pytest.fixture
def above_fd_setsize():
    """Returns a function that moves a socket to a descriptor numbered
    1024 (FD_SETSIZE) or more, which select() cannot watch."""
    fcntl = pytest.importorskip('fcntl')
    resource = pytest.importorskip('resource')
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = 1024 + 64
    if soft < want:
        if hard != resource.RLIM_INFINITY and hard < want:
            pytest.skip('the open file limit cannot be raised above 1024')
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    moved = []

    def move(sock):
        high = socket.socket(
            fileno=fcntl.fcntl(sock.fileno(), fcntl.F_DUPFD, 1024))
        sock.close()
        moved.append(high)
        return high

    yield move
    for sock in moved:
        sock.close()
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_runonce_relays_on_descriptors_above_fd_setsize(above_fd_setsize):
    """With about a thousand connections open, new sockets get descriptors
    that select() rejects with "filedescriptor out of range in select()"."""
    a, a_peer = socket.socketpair()
    b, b_peer = socket.socketpair()
    a, b = above_fd_setsize(a), above_fd_setsize(b)
    assert min(a.fileno(), b.fileno()) >= 1024
    handlers = [ssnet.Proxy(ssnet.SockWrapper(a, a), ssnet.SockWrapper(b, b))]
    try:
        a_peer.sendall(b'ping')
        ssnet.runonce(handlers, Mock(fullness=0, too_full=False))
        b_peer.settimeout(5)
        assert b_peer.recv(16) == b'ping'
    finally:
        a_peer.close()
        b_peer.close()


def test_mux_callback_on_descriptors_above_fd_setsize(above_fd_setsize):
    def packet(cmd, data):
        return struct.pack('!ccHHH', b'S', b'S', 0, cmd, len(data)) + data

    ours, theirs = socket.socketpair()
    ours = above_fd_setsize(ours)
    assert ours.fileno() >= 1024
    mux = ssnet.Mux(ours.makefile('rb', buffering=0),
                    ours.makefile('wb', buffering=0))
    try:
        theirs.sendall(packet(ssnet.CMD_PING, b'hi'))
        mux.callback(None)
        while mux.outbuf:
            mux.callback(None)
        theirs.settimeout(5)
        expected = (packet(ssnet.CMD_PING, b'chicken') +
                    packet(ssnet.CMD_PONG, b'hi'))
        received = b''
        while len(received) < len(expected):
            received += theirs.recv(64)
        assert received == expected
    finally:
        theirs.close()


def test_select_above_fd_setsize_reports_what_select_does(above_fd_setsize):
    a, b = socket.socketpair()
    a, b = above_fd_setsize(a), above_fd_setsize(b)
    # two objects on one descriptor, like the client's mux rfile and wfile
    rfile = a.makefile('rb', buffering=0)
    wfile = a.makefile('wb', buffering=0)
    with pytest.raises(ValueError):
        select.select([a], [], [], 0)
    assert ssnet._select([rfile, b], [wfile], [], 0) == ([], [wfile], [])
    b.send(b'x')
    assert ssnet._select([rfile, b], [wfile], [], 0) == ([rfile], [wfile], [])
    assert ssnet._select([rfile, a.fileno()], [], [], 0) == \
        ([rfile, a.fileno()], [], [])
    a.recv(1)
    assert ssnet._select([rfile], [], [], 0.01) == ([], [], [])
    b.close()  # end of file counts as readable
    assert ssnet._select([rfile], [], []) == ([rfile], [], [])


def test_select_without_poll_still_refuses_high_descriptors(
        above_fd_setsize, monkeypatch):
    a, b = socket.socketpair()
    a = above_fd_setsize(a)
    b.close()
    monkeypatch.delattr(select, 'poll', raising=False)
    with pytest.raises(ValueError):
        ssnet._select([a], [], [], 0)


def test_proxy_reads_an_idle_connection_only_when_it_is_ready():
    """Every Proxy wakes up when the shared mux is ready. Reading each idle
    connection then costs a recv() per connection per round."""
    class RecvCounter(socket.socket):
        recvs = 0

        def recv(self, *args):
            self.recvs += 1
            return super().recv(*args)

    def packet(channel, cmd, data):
        return struct.pack('!ccHHH', b'S', b'S', channel, cmd,
                           len(data)) + data

    ours, theirs = socket.socketpair()
    conn, peer = socket.socketpair()
    conn = RecvCounter(fileno=conn.detach())
    mux = ssnet.Mux(ours.makefile('rb', buffering=0),
                    ours.makefile('wb', buffering=0))
    handlers = [mux, ssnet.Proxy(ssnet.SockWrapper(conn, conn),
                                 ssnet.MuxWrapper(mux, 1))]
    sent = b''

    def mux_sent(expected):
        nonlocal sent
        while mux.outbuf:
            mux.flush()
        theirs.settimeout(5)
        try:
            while expected not in sent:
                chunk = theirs.recv(4096)
                if not chunk:
                    break
                sent += chunk
        except socket.timeout:
            pass
        return expected in sent

    try:
        theirs.sendall(packet(0, ssnet.CMD_PING, b'hi'))
        ssnet.runonce(handlers, mux)
        assert conn.recvs == 0

        # data and EOF on the connection are still relayed
        peer.sendall(b'data')
        ssnet.runonce(handlers, mux)
        assert mux_sent(packet(1, ssnet.CMD_TCP_DATA, b'data'))
        peer.close()
        ssnet.runonce(handlers, mux)
        assert mux_sent(packet(1, ssnet.CMD_TCP_EOF, b''))
    finally:
        for s in (mux.rfile, mux.wfile, ours, theirs, conn, peer):
            s.close()


@pytest.mark.parametrize('verbose', [0, 2])
def test_runonce_builds_debug_strings_only_when_logging(
        verbose, monkeypatch, capsys):
    """Formatting the descriptor lists costs time on every round."""
    monkeypatch.setattr('sshuttle.helpers.verbose', verbose)
    fds = Mock(wraps=ssnet._fds)
    monkeypatch.setattr(ssnet, '_fds', fds)
    a, a_peer = socket.socketpair()
    b, b_peer = socket.socketpair()
    handlers = [ssnet.Proxy(ssnet.SockWrapper(a, a), ssnet.SockWrapper(b, b))]
    try:
        a_peer.sendall(b'ping')
        ssnet.runonce(handlers, Mock(fullness=0, too_full=False))
        logged = capsys.readouterr().err
        if verbose:
            assert fds.called
            assert 'Waiting: 1 r=' in logged and 'Ready: 1 r=' in logged
        else:
            assert not fds.called
            assert logged == ''
    finally:
        for s in (a, a_peer, b, b_peer):
            s.close()


@pytest.mark.parametrize('verbose', [0, 2, 3])
def test_mux_debug_messages_follow_the_verbosity(verbose, monkeypatch, capsys):
    monkeypatch.setattr('sshuttle.helpers.verbose', verbose)
    ours, theirs = socket.socketpair()
    mux = ssnet.Mux(ours.makefile('rb', buffering=0),
                    ours.makefile('wb', buffering=0))  # sends a PING
    try:
        mux.flush()
        theirs.sendall(b'x')
        mux.fill()
        mux.got_packet(0, ssnet.CMD_PONG, b'')
        ours.setblocking(False)
        assert ssnet._nb_clean(ours.recv, 1) is None
        logged = capsys.readouterr().err
    finally:
        for s in (mux.rfile, mux.wfile, ours, theirs):
            s.close()
    for message, level in ((' > channel=0 cmd=PING', 2), ('mux wrote:', 2),
                           ('mux read: 1', 2), ('<  channel=0 cmd=PONG', 2),
                           ('recv: err was:', 3)):
        assert (message in logged) == (verbose >= level), message


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now


@pytest.fixture
def clocked_mux(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(ssnet, 'time', clock)
    ours, theirs = socket.socketpair()
    files = (ours.makefile('rb', buffering=0),
             ours.makefile('wb', buffering=0))
    yield ssnet.Mux(*files), clock
    for f in files + (ours, theirs):
        f.close()


def ping(mux, amount=None):
    """Send data until a PING goes out behind it and return its payload."""
    seq = mux.seq
    mux.fullness += mux.window // 4 if amount is None else amount
    mux.check_fullness()
    assert mux.seq == seq + 1
    return b'rt%d' % mux.seq


def round_trip(mux, clock, rtt):
    """Send a PING behind a full window and answer it after rtt seconds."""
    data = ping(mux, mux.window)
    clock.now += rtt
    mux.got_packet(0, ssnet.CMD_PONG, data)


def warm_up(mux, clock):
    """Measure a 50 ms idle round trip, then grow the window to 16x."""
    clock.now += 0.05
    mux.got_packet(0, ssnet.CMD_PONG, b'chicken')
    for _ in range(4):
        round_trip(mux, clock, 0.05)
    assert mux.window == 16 * ssnet.LATENCY_BUFFER_SIZE


def test_latency_window_grows_while_round_trips_stay_short(clocked_mux):
    mux, clock = clocked_mux
    assert mux.window == ssnet.LATENCY_BUFFER_SIZE
    clock.now += 0.05
    mux.got_packet(0, ssnet.CMD_PONG, b'chicken')
    assert mux.min_rtt == pytest.approx(0.05)
    assert mux.window == ssnet.LATENCY_BUFFER_SIZE  # idle PING: no change

    round_trip(mux, clock, 0.05 + ssnet.LATENCY_TARGET / 4)
    assert mux.window == 2 * ssnet.LATENCY_BUFFER_SIZE
    for _ in range(20):
        round_trip(mux, clock, 0.05)
    assert mux.window == ssnet.LATENCY_BUFFER_MAX


def test_latency_window_shrinks_when_round_trips_stay_long(clocked_mux):
    mux, clock = clocked_mux
    warm_up(mux, clock)

    slow = 0.05 + 2 * ssnet.LATENCY_TARGET
    for _ in range(ssnet.LATENCY_SAMPLES - 1):
        round_trip(mux, clock, slow)
    assert mux.window == 16 * ssnet.LATENCY_BUFFER_SIZE  # could be jitter
    round_trip(mux, clock, slow)
    assert mux.window == 8 * ssnet.LATENCY_BUFFER_SIZE
    for _ in range(3):
        round_trip(mux, clock, slow)
    assert mux.window == ssnet.LATENCY_BUFFER_SIZE
    assert mux.min_rtt == pytest.approx(0.05)


def test_latency_window_rides_out_jitter(clocked_mux):
    mux, clock = clocked_mux
    warm_up(mux, clock)
    window = mux.window
    for _ in range(10):  # two of every three round trips are too slow
        round_trip(mux, clock, 0.05 + 3 * ssnet.LATENCY_TARGET)
        round_trip(mux, clock, 0.05 + 3 * ssnet.LATENCY_TARGET)
        round_trip(mux, clock, 0.05 + ssnet.LATENCY_TARGET / 4)
    assert mux.window > window


def test_latency_window_halves_at_once_when_far_too_slow(clocked_mux):
    mux, clock = clocked_mux
    warm_up(mux, clock)
    round_trip(mux, clock, 0.05 + 5 * ssnet.LATENCY_TARGET)
    assert mux.window == 8 * ssnet.LATENCY_BUFFER_SIZE
    round_trip(mux, clock, 0.05 + 5 * ssnet.LATENCY_TARGET)
    assert mux.window == 4 * ssnet.LATENCY_BUFFER_SIZE


def test_latency_window_doubles_only_while_the_queue_stays_short(clocked_mux):
    """Doubling the window about doubles the queue: leave room for that."""
    mux, clock = clocked_mux
    warm_up(mux, clock)
    window = mux.window
    round_trip(mux, clock, 0.05 + 0.75 * ssnet.LATENCY_TARGET)
    assert mux.window == window
    round_trip(mux, clock, 0.05 + 0.4 * ssnet.LATENCY_TARGET)
    assert mux.window == 2 * window


def test_latency_window_changes_once_per_round_trip(clocked_mux):
    """Only PINGs sent behind a full new window measured it."""
    mux, clock = clocked_mux
    warm_up(mux, clock)
    window = mux.window
    mux.fullness += mux.window
    early = [ping(mux) for _ in range(3)]
    clock.now += 0.05
    mux.got_packet(0, ssnet.CMD_PONG, early[0])
    assert mux.window == 2 * window
    early = early[1:] + [ping(mux) for _ in range(3)]  # under a new window
    for data in early:
        mux.got_packet(0, ssnet.CMD_PONG, data)
    assert mux.window == 2 * window

    mux.fullness += mux.window
    late = [ping(mux) for _ in range(3)]
    clock.now += 0.05 + 5 * ssnet.LATENCY_TARGET
    for data in late:
        mux.got_packet(0, ssnet.CMD_PONG, data)
    assert mux.window == window  # halved once, not three times


def test_latency_window_takes_one_sample_per_round_trip(clocked_mux):
    """PINGs a quarter window apart share a round trip, and its jitter."""
    mux, clock = clocked_mux
    warm_up(mux, clock)
    window = mux.window
    mux.fullness += mux.window
    pings = [ping(mux) for _ in range(ssnet.LATENCY_SAMPLES)]
    clock.now += 0.05 + 2 * ssnet.LATENCY_TARGET
    for data in pings:
        mux.got_packet(0, ssnet.CMD_PONG, data)
    assert sum(rtt > 0.1 for rtt in mux.recent) == 1
    assert mux.window == window  # one slow round trip may be jitter


def test_mux_packets_shrink_with_the_window(clocked_mux):
    mux, clock = clocked_mux
    wrapper = ssnet.MuxWrapper(mux, 1)
    assert wrapper.uwrite(b'x' * 100000) == ssnet.MAX_PACKET_DATA
    mux.acked = mux.fullness
    mux.check_fullness()
    assert wrapper.uwrite(b'x' * 100000) == mux.window // 4
    mux.window = ssnet.LATENCY_BUFFER_MAX
    mux.acked = mux.fullness
    mux.check_fullness()
    assert wrapper.uwrite(b'x' * 100000) == ssnet.MAX_PACKET_DATA


def test_mux_without_latency_control_never_holds_back_data():
    """--no-latency-control: no window, even after the initial PONG."""
    ours, theirs = socket.socketpair()
    mux = ssnet.Mux(ours.makefile('rb', buffering=0),
                    ours.makefile('wb', buffering=0), latency_control=False)
    try:
        wrapper = ssnet.MuxWrapper(mux, 1)
        for _ in range(4):  # several windows before the initial PONG
            assert wrapper.uwrite(b'x' * 100000) == ssnet.MAX_PACKET_DATA
            mux.check_fullness()
        mux.got_packet(0, ssnet.CMD_PONG, b'chicken')
        mux.check_fullness()
        assert not mux.too_full
        assert wrapper.uwrite(b'x' * 100000) == ssnet.MAX_PACKET_DATA
        assert list(mux.pings) == [] and mux.seq == 0  # no PINGs sent
    finally:
        for s in (mux.rfile, mux.wfile, ours, theirs):
            s.close()


def test_latency_window_keeps_sending_while_pings_are_in_flight(clocked_mux):
    mux, clock = clocked_mux
    mux.window = 4 * ssnet.LATENCY_BUFFER_SIZE
    quarter = mux.window // 4
    pings = [ping(mux, quarter) for _ in range(3)]
    assert not mux.too_full  # three quarters in flight, PINGs unanswered
    pings.append(ping(mux, quarter))
    assert mux.too_full  # a full window (plus the PINGs) in flight
    mux.got_packet(0, ssnet.CMD_PONG, pings[0])
    assert not mux.too_full  # the first quarter arrived
    assert mux.acked >= quarter
    mux.fullness = mux.acked + mux.window + 1
    mux.check_fullness()
    assert mux.too_full
    for data in pings[1:]:
        mux.got_packet(0, ssnet.CMD_PONG, data)
    assert not mux.too_full
    assert mux.fullness > 5 * quarter  # never reset


def test_latency_window_sends_no_pings_while_idle(clocked_mux):
    mux, clock = clocked_mux
    mux.window = 1  # even a window too small to split in quarters
    mux.check_fullness()  # behind the initial PING's bytes
    queued = len(mux.outbuf)
    for _ in range(10):
        mux.check_fullness()
    assert len(mux.outbuf) == queued
    for data in list(mux.pings):
        mux.got_packet(0, ssnet.CMD_PONG, data)
    mux.check_fullness()
    assert len(mux.outbuf) == queued  # answers need no new PING
    assert not mux.too_full


def test_latency_window_ignores_unexpected_pongs(clocked_mux):
    mux, clock = clocked_mux
    mux.fullness = mux.window + 1
    mux.check_fullness()
    assert mux.too_full
    clock.now += 0.05
    mux.got_packet(0, ssnet.CMD_PONG, b'rttest')  # no such PING in flight
    mux.got_packet(0, ssnet.CMD_PONG, b'rt99')
    mux.got_packet(0, ssnet.CMD_PONG, b'chicken')
    mux.got_packet(0, ssnet.CMD_PONG, b'chicken')  # answered twice
    assert mux.window == ssnet.LATENCY_BUFFER_SIZE
    assert mux.min_rtt == pytest.approx(0.05)
    assert mux.too_full
    assert mux.acked == 0


def test_initial_pong_does_not_end_the_wait(clocked_mux):
    """The initial PING went out before the data, so it acks none of it."""
    mux, clock = clocked_mux
    mux.fullness = mux.window + 1
    mux.check_fullness()  # sent right behind the initial PING
    clock.now += 0.05
    mux.got_packet(0, ssnet.CMD_PONG, b'chicken')
    assert mux.too_full
    clock.now += 0.01
    mux.got_packet(0, ssnet.CMD_PONG, b'rt1')
    assert mux.min_rtt == pytest.approx(0.05)
    assert mux.window == 2 * ssnet.LATENCY_BUFFER_SIZE
    assert not mux.too_full


def test_mux_flush_writes_until_the_pipe_is_full():
    ours, theirs = socket.socketpair()
    ours.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
    mux = ssnet.Mux(ours.makefile('rb', buffering=0),
                    ours.makefile('wb', buffering=0))
    try:
        for i in range(200):
            mux.send(1, ssnet.CMD_TCP_DATA, bytes([i]) * 30000)
        expected = b''.join(mux.outbuf)
        mux.flush()
        # One call wrote more than one packet, and stopped at a full pipe.
        written = len(expected) - mux.amount_queued()
        assert written > 30008
        assert mux.outbuf
        theirs.settimeout(5)
        received = b''
        while mux.outbuf or len(received) < len(expected):
            received += theirs.recv(1 << 20)
            mux.flush()
        assert received == expected
    finally:
        theirs.close()


def test_mux_flush_lets_channels_refill_once_drained():
    ours, theirs = socket.socketpair()
    mux = ssnet.Mux(ours.makefile('rb', buffering=0),
                    ours.makefile('wb', buffering=0))
    wrapper = ssnet.MuxWrapper(mux, 1)
    try:
        for _ in range(3):
            mux.send(1, ssnet.CMD_TCP_DATA, b'x' * ssnet.MAX_PACKET_DATA)
        mux.pre_select([], [], [])
        assert wrapper.too_full()
        mux.flush()  # the socket buffer takes all of it
        assert not mux.outbuf
        assert not wrapper.too_full()
    finally:
        theirs.close()


def test_muxwrapper_packet_size(monkeypatch):
    a, b = socket.socketpair()
    mux = ssnet.Mux(a.makefile('rb', buffering=0),
                    a.makefile('wb', buffering=0))
    try:
        wrapper = ssnet.MuxWrapper(mux, 1)
        assert wrapper.uwrite(b'x' * 100000) == ssnet.MAX_PACKET_DATA
        monkeypatch.setattr(ssnet, 'LATENCY_BUFFER_SIZE', 4096)
        assert wrapper.uwrite(b'x' * 100000) == 4096
        assert wrapper.uwrite(b'x' * 100) == 100
        monkeypatch.setattr(ssnet, 'LATENCY_BUFFER_SIZE', -1)
        assert wrapper.uwrite(b'xy') == 1
        sizes = [len(p) - ssnet.HDR_LEN for p in mux.outbuf[1:]]
        assert sizes == [ssnet.MAX_PACKET_DATA, 4096, 100, 1]
    finally:
        a.close()
        b.close()


def test_a_busy_channel_does_not_hold_up_the_others():
    """Commit 5d13909 cut mux packets to 2 KiB because "a single busy stream
    can ruin it for everybody". With full-size packets the mux queue stays
    bounded and a quiet channel still gets through."""
    mux_end, pipe = socket.socketpair()
    pipe.setblocking(False)
    mux = ssnet.Mux(mux_end.makefile('rb', buffering=0),
                    mux_end.makefile('wb', buffering=0))
    handlers, peers = [mux], []
    for channel in (1, 2):
        s, peer = socket.socketpair()
        handlers.append(ssnet.Proxy(ssnet.SockWrapper(s, s),
                                    ssnet.MuxWrapper(mux, channel)))
        peers.append(peer)
    busy, quiet = peers
    busy.setblocking(False)
    # always readable, so runonce() returns even while the pipe is full
    tick, tick_peer = socket.socketpair()
    tick_peer.send(b't')
    handlers.append(ssnet.Handler([tick], lambda sock: None))
    # runonce() calls a Proxy once per ready socket in its socks (its own
    # socket twice, the mux's two), and each call may queue a packet.
    limit = 2 * ssnet.MAX_PACKET_DATA + 4 * (ssnet.MAX_PACKET_DATA +
                                             ssnet.HDR_LEN)
    received = b''

    def one_round():
        nonlocal received
        try:
            busy.send(b'x' * 262144)
        except BlockingIOError:
            pass
        ssnet.runonce(handlers, mux)
        assert mux.amount_queued() < limit
        try:
            received += pipe.recv(16384)  # a slow pipe
        except BlockingIOError:
            pass

    def quiet_data_arrived():
        pos = 0
        while pos + ssnet.HDR_LEN <= len(received):
            _, _, channel, cmd, n = struct.unpack(
                '!ccHHH', received[pos:pos + ssnet.HDR_LEN])
            if channel == 2 and cmd == ssnet.CMD_TCP_DATA:
                return True
            pos += ssnet.HDR_LEN + n
        return False

    try:
        for _ in range(50):
            one_round()
        assert handlers[1].wrap1.buf  # the busy channel is backlogged
        quiet.sendall(b'k')
        for _ in range(100):
            one_round()
            if quiet_data_arrived():
                break
        assert quiet_data_arrived()
    finally:
        for s in [mux_end, pipe, tick, tick_peer] + peers:
            s.close()


@pytest.mark.parametrize('seed', range(20))
def test_latency_window_never_stalls_without_a_ping_in_flight(
        clocked_mux, seed):
    """Whenever data waits, a PING behind it is on its way."""
    mux, clock = clocked_mux
    rng = random.Random(seed)
    for _ in range(2000):
        action = rng.random()
        if action < 0.5:
            mux.fullness += rng.choice(
                [0, 1, rng.randrange(mux.window), 2 * mux.window])
        elif mux.pings:
            data = rng.choice(list(mux.pings))
            clock.now += rng.choice([0.01, 0.05, 0.05, 0.2, 1.0])
            mux.got_packet(0, ssnet.CMD_PONG, data)
        mux.check_fullness()
        if mux.too_full:
            assert any(at > mux.acked for _, at in mux.pings.values())
        assert ssnet.LATENCY_BUFFER_SIZE <= mux.window
        assert mux.window <= ssnet.LATENCY_BUFFER_MAX


def test_pong_answers_ping_with_the_same_payload(clocked_mux):
    mux, clock = clocked_mux
    del mux.outbuf[:]
    mux.got_packet(0, ssnet.CMD_PING, b'rt7')
    assert mux.outbuf == [struct.pack('!ccHHH', b'S', b'S', 0,
                                      ssnet.CMD_PONG, 3) + b'rt7']
