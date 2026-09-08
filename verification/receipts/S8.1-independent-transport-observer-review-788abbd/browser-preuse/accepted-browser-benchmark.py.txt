"""Task-local headless browser measurements; no model construction or user profile."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import urllib.request

import websocket


class CDP:
    def __init__(self, url):
        self.socket = websocket.create_connection(url, timeout=60, suppress_origin=True)
        self.counter = 0
        self.events = []

    def call(self, method, params=None):
        self.counter += 1
        ident = self.counter
        self.socket.send(json.dumps({'id': ident, 'method': method, 'params': params or {}}))
        while True:
            message = json.loads(self.socket.recv())
            if message.get('id') == ident:
                if 'error' in message:
                    raise RuntimeError(message['error'])
                return message.get('result', {})
            self.events.append(message)

    def wait_event(self, method):
        for event in self.events:
            if event.get('method') == method:
                return event
        while True:
            event = json.loads(self.socket.recv())
            self.events.append(event)
            if event.get('method') == method:
                return event

    def evaluate(self, expression):
        result = self.call('Runtime.evaluate', {
            'expression': expression, 'awaitPromise': True, 'returnByValue': True,
        })
        if result.get('exceptionDetails'):
            raise RuntimeError(result['exceptionDetails'])
        return result['result'].get('value')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--html', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--browser', type=Path, required=True)
    parser.add_argument('--route', action='append', default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.html.read_bytes()
    receipt = {
        'input': str(args.html.resolve()), 'input_sha256': hashlib.sha256(raw).hexdigest(),
        'input_bytes': len(raw), 'viewport': {'width': 1440, 'height': 1000, 'deviceScaleFactor': 1},
        'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'method': 'Fresh task-local headless Chromium profile; file navigation; DOM click via CDP mouse events; click capture to second animation frame. No model run or user session.',
        'timing_limit': 'Headless fixed-engine timings; two animation frames include a paint opportunity, not a hardware presentation timestamp. Main HTML is local; all HTTP(S) page requests are explicitly blocked, including Google Fonts, so both versions use the same offline fallback-font policy. Online font/network performance is not measured.',
        'clicks': [],
    }
    process = None
    client = None
    profile_dir = None
    stderr = None
    failure = None
    try:
        profile_dir = tempfile.TemporaryDirectory(prefix="unfold-s81-browser-")
        profile = profile_dir.name
        stderr = (args.output / 'browser.stderr').open('w')
        command = [str(args.browser), '--headless', '--no-first-run',
                   '--no-default-browser-check', '--disable-background-networking',
                   '--disable-background-timer-throttling', '--disable-renderer-backgrounding',
                   '--remote-debugging-port=0', '--user-data-dir=' + profile, 'about:blank']
        receipt['command'] = command
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=stderr,
                                   start_new_session=True)
        endpoint = Path(profile) / 'DevToolsActivePort'
        deadline = time.monotonic() + 30
        while not endpoint.exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Dedicated browser did not expose its debugging endpoint')
            time.sleep(0.05)
        port = int(endpoint.read_text().splitlines()[0])
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list', timeout=10) as response:
            targets = json.load(response)
        page = next(target for target in targets if target['type'] == 'page')
        client = CDP(page['webSocketDebuggerUrl'])
        receipt['browser_version'] = client.call('Browser.getVersion')
        client.call('Page.enable')
        client.call('Runtime.enable')
        client.call('Network.enable')
        client.call('Network.setBlockedURLs', {'urls': ['http://*', 'https://*']})
        client.call('Emulation.setDeviceMetricsOverride', {
            **receipt['viewport'], 'mobile': False,
        })
        client.call('Page.addScriptToEvaluateOnNewDocument', {'source': """
            window.__s81ReadinessPromise = new Promise(resolve => {
                window.addEventListener('load', () => document.fonts.ready.then(() => {
                    requestAnimationFrame(() => requestAnimationFrame(() => {
                        const ready_ms=performance.now();
                        const nodes=Array.from(document.querySelectorAll('.uf-section-arch .uf-node'));
                        resolve({ready_ms,navigation:performance.getEntriesByType('navigation').map(x=>x.toJSON()),
                            dom_nodes:document.querySelectorAll('*').length,
                            architecture_nodes:nodes.map(n=>({id:n.getAttribute('data-id'),text:n.textContent})),
                            cards:document.querySelectorAll('.uf-card-detail').length});
                    }));
                }), {once:true});
            });
        """})
        client.events.clear()
        start = time.perf_counter()
        navigation = client.call('Page.navigate', {'url': args.html.resolve().as_uri()})
        if navigation.get('errorText'):
            raise RuntimeError(navigation['errorText'])
        client.wait_event('Page.loadEventFired')
        readiness = client.evaluate('window.__s81ReadinessPromise')
        if not readiness or not readiness['architecture_nodes']:
            raise RuntimeError('No actual architecture nodes in loaded page')
        receipt['navigation_rpc_wall_seconds'] = time.perf_counter() - start
        receipt['readiness'] = readiness
        image = client.call('Page.captureScreenshot', {'format': 'png', 'captureBeyondViewport': False})
        (args.output / 'initial.png').write_bytes(base64.b64decode(image['data']))
        for index, target_id in enumerate(args.route):
            selector = ('.uf-section-arch' if index == 0 else '.uf-inspect-panel')
            target = client.evaluate('''(() => {
                const id = ''' + json.dumps(target_id) + ''';
                const scope = ''' + ("document.querySelector('.uf-section-arch')" if index == 0 else f"document.querySelectorAll('.uf-inspect-panel')[{index - 1}]") + ''';
                const visible = n => n.getClientRects().length && getComputedStyle(n).visibility !== 'hidden';
                const matches = Array.from(scope.querySelectorAll('.uf-node')).filter(n => n.getAttribute('data-id') === id && visible(n));
                if(matches.length !== 1) throw new Error('Expected one visible occurrence: '+id+' got '+matches.length);
                const n=matches[0];n.scrollIntoView({block:'center',inline:'center'});
                const r=n.getBoundingClientRect();return {id,text:n.textContent,x:r.x+r.width/2,y:r.y+r.height/2};
            })()''')
            client.evaluate("""window.__s81Measurement = null;
                document.addEventListener('click', function(event) {
                    const start=performance.now();const clicked=event.target.closest('.uf-node');
                    requestAnimationFrame(() => requestAnimationFrame(() => {
                        window.__s81Measurement={milliseconds:performance.now()-start,clicked_id:clicked&&clicked.getAttribute('data-id'),
                            active_cards:Array.from(document.querySelectorAll('.uf-card-detail')).filter(n=>n.getClientRects().length).map(n=>({id:n.getAttribute('data-card-id'),text:n.textContent,panel_index:Array.from(document.querySelectorAll('.uf-inspect-panel')).indexOf(n.closest('.uf-inspect-panel'))})),
                            dom_nodes:document.querySelectorAll('*').length,cards:document.querySelectorAll('.uf-card-detail').length};
                    }));
                }, {capture:true,once:true});""")
            for kind in ('mousePressed', 'mouseReleased'):
                client.call('Input.dispatchMouseEvent', {'type': kind, 'x': target['x'], 'y': target['y'], 'button': 'left', 'clickCount': 1})
            measured = client.evaluate("""new Promise((resolve,reject)=>{const limit=performance.now()+5000;function check(){if(window.__s81Measurement)resolve(window.__s81Measurement);else if(performance.now()>limit)reject(new Error('Click measurement missing'));else requestAnimationFrame(check)}check()})""")
            if measured['clicked_id'] != target_id:
                raise RuntimeError('Dispatched click did not target the expected occurrence')
            if not any(card['id'] == target_id and card['panel_index'] == index for card in measured['active_cards']):
                raise RuntimeError('Expected own card did not become visible in the next panel')
            receipt['clicks'].append({'depth': index, 'selector_scope': selector, 'target': target, 'result': measured})
            image = client.call('Page.captureScreenshot', {'format': 'png', 'captureBeyondViewport': False})
            (args.output / f'click-{index:02d}.png').write_bytes(base64.b64decode(image['data']))
        receipt['exceptions'] = [event for event in client.events if event.get('method') == 'Runtime.exceptionThrown']
        receipt['network_failures'] = [event for event in client.events if event.get('method') == 'Network.loadingFailed']
        if receipt['exceptions']:
            raise RuntimeError('Unresolved runtime exception in actual page')
        receipt['status'] = 'ok'
    except BaseException as exc:
        receipt['status'] = 'failed'
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        failure = exc
    finally:
        cleanup_errors = []
        if client is not None:
            try:
                client.socket.close()
            except Exception as exc:
                cleanup_errors.append(f'socket: {exc}')
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception as exc:
                cleanup_errors.append(f'terminate: {exc}')
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            except Exception as exc:
                cleanup_errors.append(f'wait: {exc}')
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as exc:
                cleanup_errors.append(f'final group cleanup: {exc}')
        if stderr is not None:
            stderr.close()
        if profile_dir is not None:
            try:
                profile_dir.cleanup()
            except Exception as exc:
                cleanup_errors.append(f'profile: {exc}')
        if cleanup_errors:
            receipt['cleanup_errors'] = cleanup_errors
            receipt['status'] = 'failed'
            receipt.setdefault('error', 'Browser cleanup failed')
        receipt['input_after_sha256'] = hashlib.sha256(args.html.read_bytes()).hexdigest()
        if receipt['input_after_sha256'] != receipt['input_sha256']:
            receipt['status'] = 'failed'
            receipt['error'] = 'Input HTML changed during browser measurement'
        (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps({'status':receipt['status'],'output':str(args.output)}))
    if failure is not None:
        raise failure
    if receipt['status'] != 'ok':
        raise RuntimeError(receipt['error'])


if __name__ == '__main__':
    main()
