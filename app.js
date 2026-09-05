'use strict';
const workflows = {
  recommend: {
    comment: '# Know the family, but not which model to run?',
    command: 'uv run infercap check Qwen --recommend-limit 5',
    lines: [],
    note: '',
    caption: 'Example only · real discovery requires Hugging Face Hub access'
  },
  check: {
    comment: '# Start with the hardware you have.',
    command: 'uv run infercap check \\\n  --model mistralai/Mistral-7B-Instruct-v0.3',
    lines: [['PASS', 'Python, PyTorch & vLLM'], ['PASS', 'NVIDIA GPU detected'], ['PASS', 'Model architecture supported'], ['INFO', 'Model-weight memory estimated']],
    note: 'Next → review the recommended serving command.',
    caption: 'Illustrative output · results depend on your environment'
  },
  serve: {
    comment: '# Request a serving profile for your model.',
    command: 'uv run infercap check \\\n  --model mistralai/Mistral-7B-Instruct-v0.3 \\\n  --profile balanced',
    lines: [['01', 'Review the generated vllm serve command.'], ['02', 'Run that command in a separate terminal.'], ['03', 'Check the served model with --check-server.']],
    note: 'Recommendations are starting points; start the server yourself.',
    caption: 'Workflow guide · preflight does not start the server'
  },
  benchmark: {
    comment: '# With your model server running, sweep the load.',
    command: 'uv run infercap benchmark \\\n  --model mistralai/Mistral-7B-Instruct-v0.3 \\\n  --concurrency 1,4,8,16,32,64',
    lines: [['→', 'Measure output TPS, RPS & latency.'], ['→', 'Sample available GPU and vLLM telemetry.'], ['→', 'Analyze saturation across tested levels.'], ['→', 'Export JSON + PNG to benchmark_output/.']],
    note: 'Review the JSON report and generated PNG dashboard.',
    caption: 'Workflow guide · requires a running compatible endpoint'
  }
};
let activeStep = 'recommend';
const tabs = [...document.querySelectorAll('[data-step]')];
function selectStep(key) {
  activeStep = key;
  document.getElementById('recommend-demo').hidden = key !== 'recommend';
  const workflow = workflows[key];
  tabs.forEach(tab => {
    const selected = tab.dataset.step === key;
    tab.classList.toggle('active', selected);
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  document.getElementById('command-panel').setAttribute('aria-labelledby', `tab-${key}`);
  document.getElementById('terminal-comment').textContent = workflow.comment;
  document.getElementById('terminal-command').textContent = workflow.command;
  document.getElementById('terminal-caption').textContent = workflow.caption;
  const output = document.getElementById('terminal-output');
  output.replaceChildren();
  workflow.lines.forEach(([label, text]) => {
    const line = document.createElement('p');
    const badge = document.createElement('b');
    badge.textContent = `[${label}]`;
    line.append(badge, text);
    output.append(line);
  });
  const note = document.createElement('p');
  note.className = 'output-note';
  note.textContent = workflow.note;
  if (workflow.note) output.append(note);
}
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => selectStep(tab.dataset.step));
  tab.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (index + 1) % tabs.length;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = (index + tabs.length - 1) % tabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = tabs.length - 1;
    if (next === undefined) return;
    event.preventDefault();
    selectStep(tabs[next].dataset.step);
    tabs[next].focus();
  });
});
let toastTimer;
async function copyCommand(text) {
  const toast = document.getElementById('toast');
  try {
    await navigator.clipboard.writeText(text);
    toast.textContent = 'Command copied to clipboard';
  } catch {
    toast.textContent = 'Clipboard unavailable. Select and copy the command manually.';
  }
  clearTimeout(toastTimer);
  toast.classList.add('show');
  toastTimer = setTimeout(() => toast.classList.remove('show'), 3200);
}
document.querySelectorAll('[data-copy]').forEach(button => button.addEventListener('click', () => copyCommand(button.dataset.copy)));
document.getElementById('copy-workflow').addEventListener('click', () => copyCommand(workflows[activeStep].command));
selectStep(activeStep);

// Feature cards open the matching example, including for keyboard users.
document.querySelectorAll('[data-workflow]').forEach(link => {
  link.addEventListener('click', () => {
    selectStep(link.dataset.workflow);
    document.getElementById(`tab-${link.dataset.workflow}`).focus({ preventScroll: true });
  });
});
