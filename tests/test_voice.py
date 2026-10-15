import pathlib
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))
import voice_engine
from voice_engine import Priority, VoiceEngine


class FakeSpeech:
    """An event-pumped driver that can hold a real utterance open."""

    def __init__(self):
        self.started = []
        self.stopped = []
        self.callbacks = {}
        self.current = None
        self.held = set()
        self.owner = None
        self.threads = set()
        self.loop_ended = False

    def record_thread(self):
        self.threads.add(threading.get_ident())

    def setProperty(self, *args):
        self.record_thread()

    def getProperty(self, name):
        self.record_thread()
        return []

    def connect(self, event, callback):
        self.callbacks[event] = callback

    def startLoop(self, use_driver):
        self.record_thread()
        assert use_driver is False

    def say(self, message, name=None):
        self.record_thread()
        self.current = (message, name)
        self.started.append(message)
        self.callbacks['started-utterance'](name)

    def iterate(self):
        self.record_thread()
        if self.current and self.current[0] not in self.held:
            self.current = None

    def isBusy(self):
        return self.current is not None

    def stop(self):
        self.record_thread()
        if self.current:
            self.stopped.append(self.current[0])
        self.current = None

    def endLoop(self):
        self.record_thread()
        self.loop_ended = True


class VoiceTests(unittest.TestCase):
    def setUp(self):
        self.driver = FakeSpeech()
        self.patch = patch.object(voice_engine, 'pyttsx3', SimpleNamespace(init=lambda: self.driver))
        self.patch.start()
        self.spoken = []
        self.voice = VoiceEngine(cooldown_seconds=30, on_spoken=self.spoken.append)
        self.voice.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.voice.shutdown)

    def wait_for(self, predicate):
        deadline = time.monotonic() + 2
        while not predicate():
            if time.monotonic() >= deadline:
                self.fail('Speech worker did not reach expected state')
            time.sleep(0.002)

    def block(self, priority=Priority.CRITICAL):
        self.driver.held.add('blocking')
        self.assertTrue(self.voice.say('blocking', priority))
        self.wait_for(lambda: 'blocking' in self.driver.started)

    def release(self):
        self.driver.held.discard('blocking')

    def test_priority_order_and_fifo_within_rank(self):
        self.block()
        for message, rank in [('normal', Priority.NORMAL), ('sector one', Priority.TIMING),
                              ('gap', Priority.RACE), ('pit', Priority.STRATEGY),
                              ('sector two', Priority.TIMING)]:
            self.assertTrue(self.voice.say(message, rank))
        self.release()
        self.wait_for(lambda: len(self.spoken) == 6)
        self.assertEqual(self.spoken, ['blocking', 'pit', 'gap', 'sector one', 'sector two', 'normal'])

    def test_expired_pending_callout_is_never_spoken(self):
        self.block()
        self.assertTrue(self.voice.say('old braking point', ttl=0.02))
        time.sleep(0.04)
        self.voice.say('still relevant', ttl=10)
        self.release()
        self.wait_for(lambda: 'still relevant' in self.spoken)
        self.assertNotIn('old braking point', self.spoken)

    def test_pending_topic_is_replaced_and_spoken_topic_uses_cooldown(self):
        self.block()
        self.assertTrue(self.voice.say('gap three seconds', key='gap'))
        self.assertTrue(self.voice.say('gap two seconds', key='gap'))
        self.release()
        self.wait_for(lambda: 'gap two seconds' in self.spoken)
        self.assertNotIn('gap three seconds', self.spoken)
        self.assertFalse(self.voice.say('gap one second', key='gap'))
        self.assertTrue(self.voice.say('gap one second', key='gap', cooldown=0))
        self.wait_for(lambda: 'gap one second' in self.spoken)

    def test_queue_is_bounded_and_keeps_urgent_callout_when_full(self):
        self.block()
        accepted = []
        for number in range(100):
            accepted.append(self.voice.say(f'low {number}', Priority.LOW))
        self.assertEqual(sum(accepted), 64)
        self.assertTrue(self.voice.say('pit window', Priority.STRATEGY))
        self.release()
        self.wait_for(lambda: len(self.spoken) == 65)
        self.voice.shutdown()
        self.assertEqual(len(self.spoken), 65)
        self.assertEqual(self.spoken[1], 'pit window')
        self.assertNotIn('low 63', self.spoken)

    def test_urgent_callout_preempts_on_worker_thread(self):
        for rank in (Priority.CRITICAL, Priority.STRATEGY, Priority.RACE, Priority.TIMING):
            with self.subTest(rank=rank):
                message = f'background {rank}'
                urgent = f'urgent {rank}'
                self.driver.held.add(message)
                self.voice.say(message, Priority.LOW)
                self.wait_for(lambda: message in self.spoken)
                self.voice.say(urgent, rank)
                self.wait_for(lambda: urgent in self.spoken)
                self.assertIn(message, self.driver.stopped)
        self.assertEqual(len(self.driver.threads), 1)
        self.assertNotIn(threading.get_ident(), self.driver.threads)

    def test_active_topic_cannot_duplicate_even_without_cooldown(self):
        self.block()
        self.assertFalse(self.voice.say('another blocking', key='blocking', cooldown=0))

    def test_immediate_bypasses_spoken_message_cooldown(self):
        self.voice.say('critical warning')
        self.wait_for(lambda: self.spoken == ['critical warning'] and not self.driver.isBusy())
        self.assertFalse(self.voice.say('critical warning'))
        self.assertTrue(self.voice.say_immediate('critical warning'))
        self.wait_for(lambda: self.spoken == ['critical warning', 'critical warning'])

    def test_same_rank_does_not_interrupt(self):
        self.block(Priority.TIMING)
        self.voice.say('next sector', Priority.TIMING)
        time.sleep(0.03)
        self.assertEqual(self.spoken, ['blocking'])
        self.release()
        self.wait_for(lambda: 'next sector' in self.spoken)

    def test_clear_cancels_active_and_pending(self):
        self.block()
        self.voice.say('discard this')
        self.voice.clear()
        self.voice.say('after clear')
        self.wait_for(lambda: 'after clear' in self.spoken)
        self.assertIn('blocking', self.driver.stopped)
        self.assertNotIn('discard this', self.spoken)

    def test_shutdown_joins_worker_and_restart_works(self):
        self.block()
        self.voice.shutdown()
        self.assertFalse(self.voice.is_running)
        self.assertTrue(self.driver.loop_ended)
        self.assertFalse(any(t.name == 'VoiceEngine' and t.is_alive() for t in threading.enumerate()))
        self.assertFalse(self.voice.say('after shutdown'))
        self.voice.start()
        self.assertTrue(self.voice.is_running)
        self.voice.say('restarted')
        self.wait_for(lambda: 'restarted' in self.spoken)


class VoiceFailureTests(unittest.TestCase):
    def test_unavailable_or_failed_driver_does_not_accept_messages(self):
        for driver in [None, SimpleNamespace(init=lambda: (_ for _ in ()).throw(RuntimeError('no driver')))]:
            with self.subTest(driver=driver), patch.object(voice_engine, 'pyttsx3', driver):
                voice = VoiceEngine()
                self.addCleanup(voice.shutdown)
                voice.start()
                self.assertFalse(voice.is_running)
                self.assertFalse(voice.say('no worker'))


if __name__ == '__main__':
    unittest.main()
