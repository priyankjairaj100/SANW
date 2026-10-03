#!/usr/bin/env python3
"""Compile the flat ACL source, rejecting unfilled results in final mode.

No historical results are read. Final mode requires a source/analysis receipt
(manuscript/paper_results_receipt.json) written after new results are audited.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'manuscript'


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft', action='store_true', help='Permit visible pending-result markers; output is explicitly a draft.')
    parser.add_argument('--check-only', action='store_true', help='Validate source and style integrity without creating a PDF.')
    args = parser.parse_args()
    provenance = json.loads((SOURCE / 'acl_style_provenance.json').read_text())
    for name in ('acl.sty', 'acl_natbib.bst'):
        if digest(SOURCE / name) != provenance['files'][name]['sha256']:
            raise SystemExit(f'Official ACL style modified: {name}')
    sources = [p for p in SOURCE.glob('*.tex') if p.name != 'official_template_example.tex']
    pending = []
    for path in sources:
        source = path.read_text()
        if '\u2014' in source or '---' in source:
            raise SystemExit(f'Em dash found in {path.name}')
        if re.search(r'\\pending\{', source):
            pending.append(path.name)
    if pending and not args.draft:
        raise SystemExit('Refusing final build with pending new results: ' + ', '.join(sorted(pending)))
    if not args.draft:
        receipt_path = SOURCE / 'paper_results_receipt.json'
        if not receipt_path.exists():
            raise SystemExit('Final build requires paper_results_receipt.json referencing audited fresh analysis.')
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('execution_kind') != 'fresh_rerun' or not receipt.get('analysis_inputs'):
            raise SystemExit('Invalid new-result receipt.')
        if receipt.get('historical_results_read') or not receipt.get('diagnostic_complete'):
            raise SystemExit('Final output requires complete fresh diagnostic and no historical inputs.')
        if receipt.get('generator_sha256') != digest(ROOT / 'scripts/generate_paper_results.py'):
            raise SystemExit('Paper generator changed; regenerate paper results.')
        for item in receipt.get('generated_outputs', []):
            path = ROOT / item['path']
            if digest(path) != item['sha256']:
                raise SystemExit(f'Generated paper asset changed: {path}; regenerate it from fresh results.')
        for item in receipt['analysis_inputs']:
            path = (ROOT / item['path']).resolve()
            if not path.is_relative_to(ROOT) or 'historical_context' in path.parts:
                raise SystemExit('New-result inputs must be fresh project outputs, not historical context.')
            if digest(path) != item['sha256']:
                raise SystemExit(f'Analysis input has changed: {path}')
    if args.check_only:
        print(json.dumps({'status': 'source_checked', 'draft': args.draft, 'pending_files': sorted(pending)}, indent=2))
        return
    missing = [tool for tool in ('pdflatex', 'bibtex') if not shutil.which(tool)]
    if missing:
        raise SystemExit('Missing TeX tools: ' + ', '.join(missing))
    env = os.environ.copy()
    probe = subprocess.run(['kpsewhich', 'pdflatex.fmt'], stdout=subprocess.PIPE, text=True)
    if not probe.stdout.strip():
        import importlib.util
        spec = importlib.util.spec_from_file_location('repair_tex_runtime', SOURCE / 'repair_tex_runtime.py')
        repair = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(repair)
        env = repair.configure()
    build = SOURCE / 'build'
    build.mkdir(exist_ok=True)
    command = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', '-file-line-error', '-output-directory=build', 'main.tex']
    commands = [command, ['bibtex', 'build/main'], command, command]
    for index, cmd in enumerate(commands):
        proc = subprocess.run(cmd, cwd=SOURCE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (build / f'command-{index + 1}.log').write_text(proc.stdout)
        if proc.returncode:
            print(proc.stdout[-6000:], file=sys.stderr)
            raise SystemExit(f'Compilation failed ({cmd[0]}), see manuscript/build/command-{index + 1}.log')
    log = (build / 'main.log').read_text(errors='replace')
    issues = [line for line in log.splitlines() if re.search(r'Overfull \\[hv]box|undefined|multiply defined', line)]
    if issues and not args.draft:
        print('\n'.join(issues), file=sys.stderr)
        raise SystemExit('Final build has unresolved TeX warnings.')
    destination = ROOT / 'output' / 'pdf'
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / ('paper-draft.pdf' if args.draft else 'acl27_relation_labels_final_rewrite_v2.pdf')
    shutil.copy2(build / 'main.pdf', output)
    if not args.draft:
        shutil.copy2(output, SOURCE / 'paper.pdf')
    metadata = {'status': 'compiled_draft' if args.draft else 'compiled_requires_visual_review',
                'output': str(output.relative_to(ROOT)), 'sha256': digest(output),
                'bytes': output.stat().st_size, 'pending_files': sorted(pending), 'tex_warnings': issues,
                'source_hashes': {p.name: digest(p) for p in sorted(sources)},
                'supporting_source_hashes': {name: digest(SOURCE / name) for name in ('references.bib', 'acl.sty', 'acl_natbib.bst')},
                'build_script_sha256': digest(pathlib.Path(__file__))}
    try:
        import fitz
        document = fitz.open(output)
        metadata['pages'] = len(document)
        metadata['limitations_pages'] = [i + 1 for i, page in enumerate(document) if 'Limitations' in page.get_text()]
        metadata['type3_fonts'] = sorted({font[3] for page in document for font in page.get_fonts() if font[2] == 'Type3'})
        if metadata['type3_fonts'] and not args.draft:
            raise SystemExit('Type 3 fonts found; repair the local TeX installation before final delivery.')
    except ImportError:
        metadata['pdf_inspection'] = 'PyMuPDF unavailable; visual review required.'
    (destination / ('draft_build_receipt.json' if args.draft else 'paper_build_receipt_v2.json')).write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
