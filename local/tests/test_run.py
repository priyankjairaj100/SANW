"""Small orchestration tests. No fitting, array loading, or outcome scoring."""
import contextlib
import ast
import argparse
import sys
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / 'run.py'
spec = importlib.util.spec_from_file_location('local_run', MODULE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class WorkflowTests(unittest.TestCase):
    def plan(self, phase):
        runner=m.Runner()
        with contextlib.redirect_stdout(io.StringIO()):
            m.build_phase(phase,runner)
        return runner.commands

    def test_dry_run_never_invokes_subprocess(self):
        with patch.object(m.subprocess,'run',side_effect=AssertionError('execution')):
            for phase in m.PHASES:
                if phase not in ('status','preflight','report'):
                    self.plan(phase)

    def test_prepare_restores_sources_before_other_assets(self):
        commands=self.plan('prepare')
        self.assertEqual(commands[0][1:],['local/restore_science.py','--execute'])
        self.assertEqual(commands[1][1],'local/assets.py')

    def test_fixed_replication_matrix(self):
        commands=self.plan('rn-replicas')
        self.assertEqual(len(commands),2)
        self.assertEqual([c[c.index('--seed')+1] for c in commands],['29','43'])
        self.assertTrue(all(c[c.index('--encoder')+1]=='rn50' for c in commands))

    def test_audit_replicas_are_four_numerical_audits(self):
        commands=self.plan('audit-replicas')
        self.assertEqual(len(commands),4)
        self.assertTrue(all(c[1]==m.SCRIPTS['audit'] for c in commands))
        self.assertFalse(any('official_results' in c[1] for c in commands))

    def test_development_locks_bind_both_encoders(self):
        for command in self.plan('lock-replicas'):
            start=command.index('--runs')+1; end=command.index('--output')
            self.assertEqual(len(command[start:end]),2)

    def test_aggregate_has_all_six_results(self):
        command=self.plan('aggregate')[0]
        results=command[command.index('--results')+1:command.index('--output')]
        self.assertEqual(len(results),6)
        self.assertEqual(set(results),{m.dev_result(e,s) for e in m.ENCODERS for s in m.SEEDS})

    def test_core_lock_has_twelve_runs_and_input_receipt(self):
        command=self.plan('core-lock')[0]
        self.assertIn('--inputs',command)
        self.assertEqual(len(command[command.index('--runs')+1:command.index('--output')]),12)

    def test_common_release_binds_lab_and_disjoint_roots(self):
        command=self.plan('release-benchmark')[0]
        self.assertEqual(command[command.index('--additional-locks')+1],m.LABL)
        self.assertEqual(len({m.COUT,m.ROUT,m.LOUT}),3)

    def test_score_core_uses_required_release_wrapper(self):
        commands=self.plan('score-benchmark')
        self.assertEqual(len(commands),18)
        core=[c for c in commands if 'score-core' in c]
        self.assertEqual(len(core),6)
        self.assertTrue(all(c[1]==m.SCRIPTS['ret'] for c in core))
        self.assertTrue(all('--release-sha256' in c for c in commands))

    def test_fresh_lock_twelve_and_release_all_states(self):
        command=self.plan('fresh-lock')[0]
        self.assertEqual(len(command[command.index('--runs')+1:command.index('--output')]),12)
        command=self.plan('fresh-release')[0]
        self.assertEqual(command[command.index('--planned-states-release')+1],m.RELEASE)

    def test_fresh_encoding_requires_preserved_lock(self):
        with patch.object(m,'require_preserved') as check:
            self.plan('fresh-encode')
            check.assert_called_once_with('fresh-inputs',False)

    def test_gate_failure_prevents_fit(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(m,'ROOT',Path(tmp)), patch.object(m.subprocess,'run') as call:
            p=Path(tmp)/m.AGG;p.parent.mkdir(parents=True);p.write_text('{"passed":false}')
            with self.assertRaises(ValueError):
                m.build_phase('no-retention',m.Runner(True))
            call.assert_not_called()

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(m,'ROOT',Path(tmp)), patch.object(m.subprocess,'run') as call:
            (Path(tmp)/'existing').mkdir()
            with self.assertRaises(FileExistsError):
                m.Runner(True).call('audit','--output','existing',output='existing')
            call.assert_not_called()

    def test_partial_fit_is_retired_intact_before_restart(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(m,'ROOT',Path(tmp)), patch.object(m.Runner,'call') as call:
            output=m.run_dir('joint','rn50',29)
            p=Path(tmp)/output;p.mkdir(parents=True);(p/'partial.txt').write_text('keep')
            m.Runner(True).fit('joint','rn50',29,'fit',[])
            retired=list((Path(tmp)/'local/retired').rglob('partial.txt'))
            self.assertEqual(len(retired),1)
            self.assertEqual(retired[0].read_text(),'keep')
            self.assertFalse(p.exists())
            call.assert_called_once()

    def test_completed_fit_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(m,'ROOT',Path(tmp)), patch.object(m.subprocess,'run') as call:
            p=Path(tmp)/m.run_dir('joint','rn50',29);p.mkdir(parents=True);(p/'completion.json').write_text('{}')
            m.Runner(True).fit('joint','rn50',29,'fit',[])
            call.assert_not_called()

    def test_source_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(m,'ROOT',Path(tmp)), patch.object(m,'source_checks',return_value=[('source.py','0'*64)]):
            (Path(tmp)/'source.py').write_text('changed')
            with self.assertRaises(ValueError):m.check_sources()

    def test_preservation_detects_changed_states(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(m,'ROOT',Path(tmp)), patch.object(m,'preservation_scope',return_value=['selected.npz']):
            (Path(tmp)/'selected.npz').write_bytes(b'exact model bytes')
            m.preserve('development',True)
            m.require_preserved('development',True)
            (Path(tmp)/'selected.npz').write_bytes(b'changed model bytes')
            with self.assertRaises(ValueError):m.require_preserved('development',True)

    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):m.path('../outside')


    def test_every_scientific_command_matches_frozen_parser(self):
        # Execute only AST statements that construct and invoke each actual parser.
        # Numerical imports and scientific function bodies are never executed.
        checked = 0
        for phase in m.PHASES:
            if phase in ('status','preflight','report'):
                continue
            for command in self.plan(phase):
                if command[1].startswith('local/'):
                    continue
                source=m.ROOT/command[1]
                module=ast.parse(source.read_text())
                main=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='main')
                statements=[]
                for statement in main.body:
                    statements.append(statement)
                    if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='parse_args' for n in ast.walk(statement)):
                        break
                else:
                    self.fail('No parser boundary: '+str(source))
                isolated=ast.Module(body=statements,type_ignores=[])
                namespace={'argparse':argparse,'Path':Path,'__file__':str(source),'__doc__':'parser test',
                           'ENCODERS':m.ENCODERS,'DATASETS':m.DATASETS,'argv':command[2:]}
                with patch.object(sys,'argv',[command[1],*command[2:]]):
                    exec(compile(isolated,str(source),'exec'),namespace)
                checked += 1
        self.assertGreaterEqual(checked,70)

    def test_future_fit_uses_new_cache(self):
        for phase in ('rn-replicas','no-retention','retrieval-only'):
            for command in self.plan(phase):
                self.assertTrue(command[command.index('--cache')+1].startswith('local/cache/'))

    def test_failed_child_logs_exit_and_output(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(m,'ROOT',Path(tmp)), patch.object(m,'environment_record',return_value={'test':True}):
            (Path(tmp)/'fake.py').write_text("print('fixture output',flush=True); raise SystemExit(7)")
            with self.assertRaises(m.subprocess.CalledProcessError):
                m.Runner(True).call('fake.py')
            receipts=list((Path(tmp)/'local/logs').glob('*.json'))
            self.assertEqual(len(receipts),1)
            receipt=json.loads(receipts[0].read_text())
            self.assertEqual(receipt['returncode'],7)
            self.assertIn('fixture output',(Path(tmp)/receipt['log']).read_text())


    def test_full_report_includes_numeric_evidence_and_excludes_inputs(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(m,'ROOT',Path(tmp)):
            include=[m.V+'full_controls/joint/selected.npz', m.V+'full_controls/joint/checkpoints/epoch_001.npz',
                     m.V+'benchmark_local_core/vit_b32/e_vil_test1000/predictions.npz',
                     m.V+'benchmark_local_core/analysis/bootstrap.npz', m.V+'benchmark_local_core/analysis/paired.npy',
                     'results/fresh_confirmation1500/vit_b32/completion.json',
                     'results/fresh_confirmation1500/vit_b32/metadata.json',
                     'data/fresh_confirmation1500/manifest.json']
            exclude=[m.V+'frozen_cache/vit_b32/image_to_text.npy',m.V+'cache/temp.npy',
                     'local/cache/rn50/text_to_image.npy','results/fresh_confirmation1500/vit_b32/features.npz',
                     'data/official_train_expansion/assets/flickr30k-images.zip',
                     'data/official_train_expansion/assets/RN50.pt']
            for name in include+exclude:
                p=Path(tmp)/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
            full=m.report_files(True); compact=m.report_files(False)
            self.assertTrue(set(include).issubset(full))
            self.assertTrue(set(exclude).isdisjoint(full))
            self.assertFalse(any(p.endswith(('.npz','.npy')) for p in compact))
            self.assertIn('data/fresh_confirmation1500/manifest.json',compact)

    def test_full_report_archive_contains_declared_evidence(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(m,'ROOT',Path(tmp)), patch.object(m,'environment_record',return_value={'fixture':True}):
            source=Path(tmp)/m.V/'example/predictions.npz'
            source.parent.mkdir(parents=True);source.write_bytes(b'evidence fixture')
            m.report(True,True)
            manifests=[p for p in (Path(tmp)/'local/reports').glob('*.tar.gz.json')]
            self.assertEqual(len(manifests),1)
            manifest=json.loads(manifests[0].read_text())
            self.assertEqual(manifest['mode'],'FULL_EVIDENCE')
            self.assertTrue(manifest['raw_prediction_and_bootstrap_arrays_included'])
            with m.tarfile.open(Path(tmp)/manifest['archive']['path'],'r:gz') as tar:
                self.assertEqual(tar.extractfile(str(source.relative_to(tmp))).read(),b'evidence fixture')
            for row in manifest['files']:
                self.assertEqual(m.digest(row['path']),row['sha256'])

    def test_full_evidence_flag_only_applies_to_report(self):
        with self.assertRaises(SystemExit),contextlib.redirect_stderr(io.StringIO()):
            m.main(['status','--full-evidence'])


    def test_executed_phase_gets_receipt(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(m,'ROOT',Path(tmp)), patch.object(m,'environment_record',return_value={'fixture':True}):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(m.main(['status','--execute']),0)
            receipts=list((Path(tmp)/'local/logs').glob('*_phase.json'))
            self.assertEqual(len(receipts),1)
            value=json.loads(receipts[0].read_text())
            self.assertEqual(value['phase'],'status')
            self.assertEqual(value['exit_code'],0)

    def test_source_closure_current_tree(self):
        self.assertEqual(m.check_sources(),217)


if __name__=='__main__':unittest.main()
