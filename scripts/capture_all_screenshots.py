"""Comprehensive Automation Script for Capturing Full-Page Screenshots of Every Tab Across All 4 Pages."""

import asyncio
import base64
import json
from pathlib import Path
import urllib.request
import websockets

SCREENSHOTS_DIR = Path("results/screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

async def send_cmd(ws, msg_id, method, params=None):
    payload = {"id": msg_id, "method": method}
    if params:
        payload["params"] = params
    await ws.send(json.dumps(payload))
    while True:
        raw = await ws.recv()
        data = json.loads(raw)
        if data.get("id") == msg_id:
            return data

async def eval_js(ws, msg_id, expression):
    return await send_cmd(ws, msg_id, "Runtime.evaluate", {
        "expression": expression,
        "awaitPromise": True,
        "returnByValue": True,
    })

async def adjust_viewport_and_capture(ws, msg_id, filename, full_page=True):
    if full_page:
        # Measure scroll height
        res = await eval_js(ws, msg_id, "Math.max(document.body.scrollHeight, document.querySelector('section.main')?.scrollHeight || 1200, 1400)")
        msg_id += 1
        h = int(res.get("result", {}).get("result", {}).get("value", 1600))
        h = max(1600, h + 150)
        await send_cmd(ws, msg_id, "Emulation.setDeviceMetricsOverride", {
            "width": 1920, "height": h, "deviceScaleFactor": 1, "mobile": False
        })
        msg_id += 1
        await asyncio.sleep(1)
        # Re-dispatch resize event to ensure Glide Data Grid adjusts to new viewport
        await eval_js(ws, msg_id, "window.dispatchEvent(new Event('resize'))")
        msg_id += 1
        await asyncio.sleep(1)
    else:
        await send_cmd(ws, msg_id, "Emulation.setDeviceMetricsOverride", {
            "width": 1920, "height": 1080, "deviceScaleFactor": 1, "mobile": False
        })
        msg_id += 1
        await asyncio.sleep(1)

    res = await send_cmd(ws, msg_id, "Page.captureScreenshot", {"format": "png"})
    msg_id += 1
    img_bytes = base64.b64decode(res["result"]["data"])
    out_path = SCREENSHOTS_DIR / filename
    with open(out_path, "wb") as f:
        f.write(img_bytes)
    print(f"[OK] Saved {filename} ({len(img_bytes):,} bytes)", flush=True)
    return msg_id

async def main():
    data = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
    page_target = next(t for t in data if t.get("type") == "page")
    ws_url = page_target["webSocketDebuggerUrl"]

    async with websockets.connect(ws_url, max_size=100_000_000) as ws:
        msg_id = 1

        # 1. Setup viewport
        msg_id += 1
        await send_cmd(ws, msg_id, "Page.enable")
        msg_id += 1
        await send_cmd(ws, msg_id, "Emulation.setDeviceMetricsOverride", {
            "width": 1920, "height": 1080, "deviceScaleFactor": 1, "mobile": False
        })

        # 2. Navigate to Streamlit App
        msg_id += 1
        await send_cmd(ws, msg_id, "Page.navigate", {"url": "http://localhost:8501/"})
        await asyncio.sleep(4)

        # Helper to click sidebar page
        async def click_page(page_name):
            nonlocal msg_id
            js = f"""
            (() => {{
                const labels = Array.from(document.querySelectorAll('div[data-testid="stRadio"] label'));
                const target = labels.find(l => l.innerText.trim() === "{page_name}");
                if (target) {{ target.click(); return true; }}
                return false;
            }})()
            """
            msg_id += 1
            res = await eval_js(ws, msg_id, js)
            await asyncio.sleep(4)
            return res.get("result", {}).get("value")

        # Helper to click tab (tabs are role="tab") and wake up canvas
        async def click_tab(tab_text):
            nonlocal msg_id
            js = f"""
            (() => {{
                const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
                const target = tabs.find(t => t.innerText.includes("{tab_text}"));
                if (target) {{ target.click(); return true; }}
                return false;
            }})()
            """
            msg_id += 1
            res = await eval_js(ws, msg_id, js)
            await asyncio.sleep(2)

            wake_js = """
            (() => {
                window.dispatchEvent(new Event('resize'));
                const dfs = Array.from(document.querySelectorAll('[data-testid="stDataFrame"]'));
                dfs.forEach(df => {
                    const canvas = df.querySelector('canvas');
                    if (canvas) {
                        const rect = canvas.getBoundingClientRect();
                        const evtMove = new MouseEvent('mousemove', {
                            bubbles: true,
                            cancelable: true,
                            clientX: rect.left + 50,
                            clientY: rect.top + 50
                        });
                        canvas.dispatchEvent(evtMove);
                        const evtClick = new MouseEvent('click', {
                            bubbles: true,
                            cancelable: true,
                            clientX: rect.left + 50,
                            clientY: rect.top + 50
                        });
                        canvas.dispatchEvent(evtClick);
                    }
                    df.scrollBy(0, 10);
                    df.scrollBy(0, -10);
                });
                window.scrollBy(0, 10);
                window.scrollBy(0, -10);
            })()
            """
            msg_id += 1
            await eval_js(ws, msg_id, wake_js)
            await asyncio.sleep(2)
            return res.get("result", {}).get("value")

        # -----------------------------------------------------------------
        # PAGE 1: EXECUTIVE OVERVIEW
        # -----------------------------------------------------------------
        print("\n--- Capturing Page 1: Overview ---", flush=True)
        await click_page("Overview")
        msg_id += 1
        await eval_js(ws, msg_id, "window.scrollTo(0, 0)")
        await asyncio.sleep(2)

        # Full page
        msg_id = await adjust_viewport_and_capture(ws, msg_id, "page1_overview.png", full_page=True)

        # Top view (viewport 1080p)
        msg_id = await adjust_viewport_and_capture(ws, msg_id, "page1_overview_top.png", full_page=False)

        # Bottom charts & footer view
        msg_id += 1
        await eval_js(ws, msg_id, "document.querySelector('section.main')?.scrollTo(0, 800)")
        await asyncio.sleep(1)
        msg_id = await adjust_viewport_and_capture(ws, msg_id, "page1_overview_charts_footer.png", full_page=False)

        # -----------------------------------------------------------------
        # PAGE 2: CLV EXPLORER
        # -----------------------------------------------------------------
        print("\n--- Capturing Page 2: CLV Explorer ---", flush=True)
        await click_page("CLV Explorer")
        msg_id += 1
        await eval_js(ws, msg_id, "document.querySelector('section.main')?.scrollTo(0, 0)")
        await asyncio.sleep(2)

        # Full page CLV initial view
        msg_id = await adjust_viewport_and_capture(ws, msg_id, "page2_clv_explorer.png", full_page=True)

        # Tab 1: P(Alive) & Lifetime Distributions (with binned heatmap)
        print("  -> Tab 1: P(Alive) & Lifetime Distributions", flush=True)
        await click_tab("P(Alive)")
        msg_id = await adjust_viewport_and_capture(ws, msg_id, "page2_clv_tab1_distributions.png", full_page=True)

        # Tab 2: CLV by Strategic Segment
        print("  -> Tab 2: CLV by Strategic Segment", flush=True)
        await click_tab("Strategic Segment")
        msg_id = await adjust_viewport_and_capture(ws, msg_id, "page2_clv_tab2_segments.png", full_page=True)

        # Tab 3: Single Customer Lookup & Uncertainty Bounds
        print("  -> Tab 3: Single Customer Lookup", flush=True)
        await click_tab("Single Customer Lookup")
        msg_id = await adjust_viewport_and_capture(ws, msg_id, "page2_clv_tab3_lookup.png", full_page=True)

        # Tab 4: Out-of-Sample Holdout Validation Tables
        print("  -> Tab 4: Holdout Validation Tables", flush=True)
        await click_tab("Holdout Validation")
        msg_id += 1
        await eval_js(ws, msg_id, "document.querySelector('section.main')?.scrollTo(0, 0)")
        await asyncio.sleep(1)
        msg_id = await adjust_viewport_and_capture(ws, msg_id, "page2_clv_tab4_holdout_side_by_side.png", full_page=False)
        msg_id = await adjust_viewport_and_capture(ws, msg_id, "page2_clv_tab4_holdout_tables.png", full_page=True)
        msg_id = await adjust_viewport_and_capture(ws, msg_id, "page2_clv_explorer_holdout_tables.png", full_page=True)

        # -----------------------------------------------------------------
        # PAGE 3: CAUSAL IMPACT & EXPERIMENTATION STUDIO
        # -----------------------------------------------------------------
        print("\n--- Capturing Page 3: Causal Impact ---", flush=True)
        await click_page("Causal Impact")
        msg_id += 1
        await eval_js(ws, msg_id, "document.querySelector('section.main')?.scrollTo(0, 0)")
        await asyncio.sleep(2)

        # Full page Causal Impact
        msg_id = await adjust_viewport_and_capture(ws, msg_id, "page3_causal_impact.png", full_page=True)
        msg_id = await adjust_viewport_and_capture(ws, msg_id, "page3_causal_impact_top.png", full_page=False)

        # Tab 1: Event-Study Dynamics
        print("  -> Tab 1: Event-Study Dynamics", flush=True)
        await click_tab("Event-Study")
        msg_id = await adjust_viewport_and_capture(ws, msg_id, "page3_causal_tab1_event_study.png", full_page=True)

        # Tab 2: ATT Estimates vs. Truth Benchmark
        print("  -> Tab 2: ATT Estimates vs. Truth Benchmark", flush=True)
        await click_tab("ATT Estimates")
        msg_id = await adjust_viewport_and_capture(ws, msg_id, "page3_causal_tab2_att_benchmark.png", full_page=True)

        # Tab 3: Q5 Divergence Gap & Deciles
        print("  -> Tab 3: Q5 Divergence Gap & Deciles", flush=True)
        await click_tab("Q5 Divergence")
        msg_id = await adjust_viewport_and_capture(ws, msg_id, "page3_causal_tab3_q5_divergence.png", full_page=True)
        msg_id = await adjust_viewport_and_capture(ws, msg_id, "page3_causal_impact_q5_gap.png", full_page=True)

        # Tab 4: Monte Carlo Coverage & Power (50 Runs)
        print("  -> Tab 4: Monte Carlo Coverage & Power", flush=True)
        await click_tab("Monte Carlo")
        msg_id = await adjust_viewport_and_capture(ws, msg_id, "page3_causal_tab4_monte_carlo.png", full_page=True)

        # -----------------------------------------------------------------
        # PAGE 4: POLICY SIMULATOR
        # -----------------------------------------------------------------
        print("\n--- Capturing Page 4: Policy Simulator ---", flush=True)
        await click_page("Policy Simulator")
        msg_id += 1
        await eval_js(ws, msg_id, "document.querySelector('section.main')?.scrollTo(0, 0)")
        await asyncio.sleep(2)

        # Full page Policy Simulator
        msg_id = await adjust_viewport_and_capture(ws, msg_id, "page4_policy_simulator.png", full_page=True)

        # Top section with renamed green box and neutral active share metric
        msg_id = await adjust_viewport_and_capture(ws, msg_id, "page4_policy_simulator_top.png", full_page=False)

        # Guardrail section
        msg_id += 1
        await eval_js(ws, msg_id, "document.querySelector('section.main')?.scrollTo(0, 450)")
        await asyncio.sleep(1)
        msg_id = await adjust_viewport_and_capture(ws, msg_id, "page4_policy_simulator_guardrails.png", full_page=False)

        # Bottom section: Band chart & sensitivity curve
        msg_id += 1
        await eval_js(ws, msg_id, "document.querySelector('section.main')?.scrollTo(0, 1100)")
        await asyncio.sleep(1)
        msg_id = await adjust_viewport_and_capture(ws, msg_id, "page4_policy_simulator_charts.png", full_page=False)
        msg_id = await adjust_viewport_and_capture(ws, msg_id, "page4_policy_simulator_sensitivity.png", full_page=False)

        print("\n[SUCCESS] Captured all screenshots across all tabs and pages into results/screenshots/!", flush=True)

asyncio.run(main())
