from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cere_contracts import ContractError
from reference_model import MemoryLedger,SessionStore,RequestLedger,should_park

class MemoryTests(unittest.TestCase):
    def test_admit_and_complete(self):
        m=MemoryLedger(100,10);m.admit('a',80,1);self.assertEqual(m.used,80)
        m.acknowledge_completion('a',m.entries['a'].ticket);self.assertEqual(m.used,0)
    def test_atomic_failure(self):
        m=MemoryLedger(100,10);m.admit('a',80,1)
        with self.assertRaises(ContractError):m.admit('b',11,1)
        self.assertEqual(m.used,80);self.assertNotIn('b',m.entries)
    def test_cancel_keeps_reservation(self):
        m=MemoryLedger(100);m.admit('a',80,2);m.request_cancel('a',m.entries['a'].ticket)
        self.assertEqual(m.used,80)
        with self.assertRaises(ContractError):m.admit('b',21,2)
    def test_stale_release(self):
        m=MemoryLedger(100);m.admit('a',80,2)
        with self.assertRaises(ContractError):m.acknowledge_completion('a',1)
        self.assertEqual(m.used,80)
    def test_double_release(self):
        m=MemoryLedger(100);ticket=m.admit('a',80,1);m.acknowledge_completion('a',ticket)
        with self.assertRaises(ContractError):m.acknowledge_completion('a',ticket)
    def test_invalid_capacity(self):
        for args in ((-1,0),(1,2),(True,0),(0,0)):
            with self.subTest(args=args),self.assertRaises(ContractError):MemoryLedger(*args)
    def test_no_negative_allocation(self):
        m=MemoryLedger(100)
        with self.assertRaises(ContractError):m.admit('a',-1,1)

class SessionTests(unittest.TestCase):
    def setUp(self):
        self.s=SessionStore();self.s.create('scope-a','parent',[1,2,3],'model-input-profile')
    def test_fork_shares_immutable_prefix(self):
        self.s.fork('scope-a','parent','child',0)
        self.assertIs(self.s.get('scope-a','parent').prefix,self.s.get('scope-a','child').prefix)
    def test_child_write_preserves_parent(self):
        self.s.fork('scope-a','parent','child',0)
        self.s.append('scope-a','child',0,[4],'model-input-profile')
        self.assertEqual(self.s.tokens('scope-a','parent'),(1,2,3))
        self.assertEqual(self.s.tokens('scope-a','child'),(1,2,3,4))
    def test_isolation(self):
        with self.assertRaises(ContractError):self.s.get('scope-b','parent')
    def test_revision_conflict(self):
        self.s.append('scope-a','parent',0,[4],'model-input-profile')
        with self.assertRaises(ContractError):self.s.append('scope-a','parent',0,[5],'model-input-profile')
        self.assertEqual(self.s.tokens('scope-a','parent'),(1,2,3,4))
    def test_fingerprint_mismatch(self):
        with self.assertRaises(ContractError):self.s.append('scope-a','parent',0,[4],'changed-template')
    def test_stale_fork(self):
        with self.assertRaises(ContractError):self.s.fork('scope-a','parent','child',1)
    def test_invalid_token_no_mutation(self):
        with self.assertRaises(ContractError):self.s.append('scope-a','parent',0,[-1],'model-input-profile')
        self.assertEqual(self.s.get('scope-a','parent').revision,0)

class SessionOwnerTests(unittest.TestCase):
    P='model-input-profile'
    def setUp(self):
        self.s=SessionStore();self.s.create('scope-a','parent',[1,2,3],self.P)
    def test_first_pin_is_one(self):
        self.assertEqual(self.s.get('scope-a','parent').owner_generation,0)
        self.assertEqual(self.s.pin('scope-a','parent','w1'),1)
        self.assertEqual(self.s.get('scope-a','parent').owner,'w1')
    def test_repin_other_worker_bumps(self):
        self.s.pin('scope-a','parent','w1')
        self.assertEqual(self.s.pin('scope-a','parent','w2'),2)
    def test_same_worker_pin_no_bump(self):
        self.s.pin('scope-a','parent','w1')
        self.assertEqual(self.s.pin('scope-a','parent','w1'),1)
    def test_migrate_cas(self):
        self.s.pin('scope-a','parent','w1')
        self.assertEqual(self.s.migrate('scope-a','parent','w2',1),2)
        with self.assertRaises(ContractError):self.s.migrate('scope-a','parent','w3',1)
        s=self.s.get('scope-a','parent');self.assertEqual((s.owner,s.owner_generation),('w2',2))
    def test_migrate_to_current_owner_refused(self):
        self.s.pin('scope-a','parent','w1')
        with self.assertRaises(ContractError):self.s.migrate('scope-a','parent','w1',1)
        self.assertEqual(self.s.get('scope-a','parent').owner_generation,1)
    def test_migrate_invalid_worker_refused(self):
        with self.assertRaises(ContractError):self.s.migrate('scope-a','parent','',0)
    def test_fork_inherits_owner(self):
        self.s.pin('scope-a','parent','w1');self.s.migrate('scope-a','parent','w2',1)
        self.s.fork('scope-a','parent','child',0)
        c=self.s.get('scope-a','child');self.assertEqual((c.owner,c.owner_generation),('w2',1))
    def test_fork_without_owner(self):
        self.s.fork('scope-a','parent','child',0)
        c=self.s.get('scope-a','child');self.assertEqual((c.owner,c.owner_generation),(None,0))
    def test_stale_held_generation_append_refused(self):
        self.s.pin('scope-a','parent','w1');self.s.migrate('scope-a','parent','w2',1)
        with self.assertRaises(ContractError):self.s.append('scope-a','parent',0,[4],self.P,held_generation=1)
        self.assertEqual(self.s.get('scope-a','parent').revision,0)
        self.assertEqual(self.s.tokens('scope-a','parent'),(1,2,3))
    def test_current_held_generation_append(self):
        self.s.pin('scope-a','parent','w1')
        self.assertEqual(self.s.append('scope-a','parent',0,[4],self.P,held_generation=1),1)
        self.assertEqual(self.s.tokens('scope-a','parent'),(1,2,3,4))

class RequestTests(unittest.TestCase):
    def setUp(self):self.r=RequestLedger();self.r.accept('a','id','digest')
    def test_idempotency(self):
        self.assertIs(self.r.accept('a','id','digest'),self.r.get('a','id'))
    def test_payload_conflict(self):
        with self.assertRaises(ContractError):self.r.accept('a','id','different')
    def test_scope_independence(self):
        self.r.accept('b','id','different');self.assertEqual(len(self.r.requests),2)
    def test_cancel_queued(self):
        self.assertEqual(self.r.cancel('a','id'),'cancelled')
        self.assertEqual(self.r.cancel('a','id'),'cancelled')
        self.assertEqual(self.r.get('a','id').terminal_count,1)
    def test_cancel_running_ack(self):
        self.r.start('a','id',1);self.assertEqual(self.r.cancel('a','id'),'cancel_pending')
        self.assertEqual(self.r.get('a','id').terminal_count,0)
        self.r.finish('a','id',self.r.get('a','id').ticket,'cancelled');self.assertEqual(self.r.get('a','id').terminal_count,1)
    def test_finish_race_with_cancel(self):
        self.r.start('a','id',1);self.r.cancel('a','id');self.r.finish('a','id',self.r.get('a','id').ticket,'completed')
        self.assertEqual(self.r.get('a','id').state,'completed')
    def test_stale_epoch(self):
        self.r.start('a','id',2)
        with self.assertRaises(ContractError):self.r.finish('a','id',1,'completed')
        self.assertEqual(self.r.get('a','id').state,'running')
    def test_duplicate_completion(self):
        self.r.start('a','id',1);self.r.finish('a','id',self.r.get('a','id').ticket,'completed')
        with self.assertRaises(ContractError):self.r.finish('a','id',self.r.get('a','id').ticket,'completed')
        self.assertEqual(self.r.get('a','id').terminal_count,1)
    def test_start_twice(self):
        self.r.start('a','id',1)
        with self.assertRaises(ContractError):self.r.start('a','id',1)
    def test_tool_wait_not_automatic_spill(self):
        self.assertFalse(should_park(wait_us=100,spill_us=100,restore_us=100,memory_pressure=True))
        self.assertFalse(should_park(wait_us=1000,spill_us=100,restore_us=100,memory_pressure=False))
        self.assertTrue(should_park(wait_us=1000,spill_us=100,restore_us=100,memory_pressure=True))
if __name__=='__main__':unittest.main()
