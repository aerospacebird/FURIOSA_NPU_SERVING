
# ============================================================
# CRAZY AI FURIOSA NPU INFERENCE LAB V1.1
#
# Hardware : FURIOSA NPU
# Backend  : Furiosa-LLM
# Model    : Meta-Llama-3.1-8B-Instruct-FP8-dynamic
# API      : OpenAI Compatible API
#
# Modules:
# 1. NPU Server Connection
# 2. Model Discovery
# 3. Server Diagnostics (urllib)
# 4. Aviation AI Inference
# 5. Transformer Inference
# 6. Python Code Generation
# 7. Streaming Inference
# 8. Repeated Inference Benchmark
# 9. JSON Result Export
#
# No requests dependency.
# ============================================================

import json
import time
import statistics
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

from openai import OpenAI


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_URL = "http://127.0.0.1:8000/v1"
API_KEY = "EMPTY"

OUTPUT_DIR = Path("./furiosa_experiment_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=180.0
)


# ============================================================
# 2. DISPLAY
# ============================================================

def section(title):

    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ============================================================
# 3. SERVER DIAGNOSTICS
#    Python Standard Library: urllib
# ============================================================

def server_diagnostics():

    section("01. FURIOSA NPU SERVER DIAGNOSTICS")

    server_url = BASE_URL.rsplit("/v1", 1)[0]

    for endpoint in ["/version", "/metrics"]:

        url = server_url + endpoint

        try:

            with urlopen(url, timeout=5) as response:

                content = response.read().decode(
                    "utf-8",
                    errors="replace"
                )

                print(
                    f"\n{endpoint}: HTTP {response.status}"
                )

                if endpoint == "/version":
                    print(content[:1500])

                elif endpoint == "/metrics":
                    print(content[:2000])

        except HTTPError as e:

            print(
                f"\n{endpoint}: HTTP ERROR {e.code}"
            )

        except URLError as e:

            print(
                f"\n{endpoint}: UNAVAILABLE - {e.reason}"
            )

        except Exception as e:

            print(
                f"\n{endpoint}: ERROR - {e}"
            )


# ============================================================
# 4. NPU CONNECTION AND MODEL DISCOVERY
# ============================================================

def connect_npu():

    section("02. FURIOSA NPU SERVER CONNECTION")

    start = time.perf_counter()

    models = client.models.list()

    latency = time.perf_counter() - start

    if not models.data:

        raise RuntimeError(
            "No models available on Furiosa server."
        )

    model_id = models.data[0].id

    print("[PASS] API Connection")

    print(f"Server       : {BASE_URL}")
    print(f"API Latency  : {latency:.4f} sec")
    print(f"Model ID     : {model_id}")

    print("\nAvailable Models:")

    for model in models.data:

        print(" -", model.id)

    server_diagnostics()

    return model_id


# ============================================================
# 5. CORE INFERENCE ENGINE
# ============================================================

def run_inference(
    model_id,
    prompt,
    system_prompt="You are CRAZY AI, a technical AI assistant.",
    max_tokens=600
):

    start = time.perf_counter()

    response = client.chat.completions.create(

        model=model_id,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2,
        max_tokens=max_tokens
    )

    elapsed = time.perf_counter() - start

    answer = (
        response.choices[0].message.content or ""
    )

    usage = response.usage

    prompt_tokens = (
        usage.prompt_tokens if usage else None
    )

    completion_tokens = (
        usage.completion_tokens if usage else None
    )

    tps = (
        completion_tokens / elapsed
        if completion_tokens is not None and elapsed > 0
        else None
    )

    result = {

        "prompt": prompt,
        "answer": answer,
        "latency_seconds": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "end_to_end_output_tps": tps,
        "model": model_id,
        "timestamp": datetime.now().isoformat()

    }

    print("\n[INFERENCE RESULT]")
    print("-" * 60)
    print(answer)

    print("\n[PERFORMANCE]")
    print(f"Latency       : {elapsed:.4f} sec")
    print(f"Input Tokens  : {prompt_tokens}")
    print(f"Output Tokens : {completion_tokens}")
    print(f"End-to-end TPS: {tps}")

    return result


# ============================================================
# 6. EXPERIMENT A: AVIATION AI
# ============================================================

def experiment_aviation(model_id):

    section("03. EXPERIMENT A - AVIATION AI")

    prompt = """
Research Topic:

B737NG aircraft Flight Management System (FMS)
and integration of an external FURIOSA NPU AI computer.

Analyze:

1. B737NG FMS architecture
2. FMC and navigation data interfaces
3. External AI computing architecture
4. NPU inference applications
5. Potential latency and reliability issues
6. Cybersecurity and data isolation
7. DO-178C and DO-254 considerations
8. Ground-based validation strategy

Distinguish verified facts from engineering proposals.

Do not propose direct modification of operational FMS,
flight-control systems or navigation data.
"""

    return run_inference(

        model_id=model_id,
        prompt=prompt,

        system_prompt=(
            "You are an experienced aviation systems "
            "engineering research assistant. "
            "Use precise technical terminology. "
            "Never invent aircraft specifications."
        ),

        max_tokens=800
    )


# ============================================================
# 7. EXPERIMENT B: TRANSFORMER ENGINEERING
# ============================================================

def experiment_transformer(model_id):

    section("04. EXPERIMENT B - TRANSFORMER AI")

    prompt = """
Explain the original Transformer architecture.

Analyze:

1. Token Embedding
2. Positional Encoding
3. Multi-Head Attention
4. Scaled Dot-Product Attention
5. Feed Forward Network
6. Residual Connections
7. Layer Normalization

Explain how FP8 quantization and NPU acceleration
affect Transformer inference.

Include mathematical equations where appropriate.
"""

    return run_inference(

        model_id=model_id,
        prompt=prompt,

        max_tokens=800
    )


# ============================================================
# 8. EXPERIMENT C: PYTHON CODE GENERATION
# ============================================================

def experiment_code(model_id):

    section("05. EXPERIMENT C - PYTHON CODE GENERATION")

    prompt = """
Write a complete PyTorch implementation of
Scaled Dot-Product Attention.

Requirements:

1. Use torch.nn.Module.
2. Support batch-first tensors.
3. Include tensor shape comments.
4. Include mathematical explanation.
5. Include a numerical test.
6. Explain potential NPU compatibility issues.

Do not claim that generated code has been executed.
"""

    return run_inference(

        model_id=model_id,
        prompt=prompt,

        system_prompt=(
            "You are a senior PyTorch and AI compiler engineer."
        ),

        max_tokens=1000
    )


# ============================================================
# 9. EXPERIMENT D: STREAMING INFERENCE
# ============================================================

def experiment_streaming(model_id):

    section("06. EXPERIMENT D - STREAMING INFERENCE")

    prompt = """
Explain how a Neural Processing Unit accelerates
Large Language Model inference.

Include:

1. Matrix multiplication
2. Attention
3. Memory bandwidth
4. Quantization
5. Token generation
"""

    start = time.perf_counter()

    first_content_time = None
    chunks = []

    stream = client.chat.completions.create(

        model=model_id,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        max_tokens=500,
        temperature=0.2,
        stream=True
    )

    print("\n[STREAMING OUTPUT]\n")

    for chunk in stream:

        content = chunk.choices[0].delta.content

        if content:

            if first_content_time is None:

                first_content_time = time.perf_counter()

            chunks.append(content)

            print(
                content,
                end="",
                flush=True
            )

    end = time.perf_counter()

    answer = "".join(chunks)

    total_time = end - start

    ttft = (

        first_content_time - start

        if first_content_time is not None

        else None

    )

    print("\n")

    print("[STREAM METRICS]")
    print(f"Total Time : {total_time:.4f} sec")
    print(f"TTFT       : {ttft}")
    print(f"Characters : {len(answer)}")

    return {

        "answer": answer,
        "total_latency_seconds": total_time,
        "time_to_first_content_seconds": ttft,
        "characters": len(answer)

    }


# ============================================================
# 10. EXPERIMENT E: NPU BENCHMARK
# ============================================================

def experiment_benchmark(model_id, iterations=5):

    section("07. EXPERIMENT E - NPU BENCHMARK")

    prompt = (
        "Explain Multi-Head Attention in a Transformer "
        "and describe its computational complexity."
    )

    latencies = []
    token_counts = []

    for i in range(iterations):

        print(f"\nTrial {i + 1}/{iterations}")

        start = time.perf_counter()

        response = client.chat.completions.create(

            model=model_id,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            max_tokens=300,
            temperature=0
        )

        elapsed = time.perf_counter() - start

        usage = response.usage

        tokens = (
            usage.completion_tokens
            if usage else None
        )

        latencies.append(elapsed)

        if tokens is not None:

            token_counts.append(tokens)

        print(f"Latency: {elapsed:.4f} sec")
        print(f"Tokens : {tokens}")

    result = {

        "iterations": iterations,

        "average_latency_seconds": (
            statistics.mean(latencies)
        ),

        "min_latency_seconds": min(latencies),

        "max_latency_seconds": max(latencies),

        "latencies": latencies,

        "completion_tokens": token_counts,

        "average_end_to_end_tps": (

            sum(token_counts) / sum(latencies)

            if token_counts

            else None

        )

    }

    print("\n[BENCHMARK SUMMARY]")

    print(
        "Average Latency:",
        round(
            result["average_latency_seconds"],
            4
        )
    )

    print(
        "Minimum Latency:",
        round(min(latencies), 4)
    )

    print(
        "Maximum Latency:",
        round(max(latencies), 4)
    )

    print(
        "Aggregate TPS:",
        result["average_end_to_end_tps"]
    )

    return result


# ============================================================
# 11. EXPERIMENT F: INTERACTIVE CHAT
# ============================================================

def interactive_chat(model_id):

    section("08. INTERACTIVE FURIOSA NPU CHAT")

    print("Enter your questions.")
    print("Type 'exit' to terminate.")

    while True:

        question = input("\nYOU: ").strip()

        if question.lower() in ["exit", "quit"]:

            break

        if not question:

            continue

        try:

            result = run_inference(

                model_id=model_id,

                prompt=question,

                system_prompt=(
                    "You are CRAZY AI, a highly capable "
                    "technical research assistant. "
                    "Provide accurate and structured answers. "
                    "Distinguish facts from assumptions."
                ),

                max_tokens=1000
            )

        except Exception as e:

            print("\n[INFERENCE ERROR]", repr(e))


# ============================================================
# 12. MAIN EXPERIMENT
# ============================================================

def main():

    experiment_start = time.perf_counter()

    results = {

        "platform": "CRAZY AI FURIOSA NPU LAB V1.1",

        "timestamp": datetime.now().isoformat(),

        "experiments": {}

    }

    try:

        model_id = connect_npu()

        results["model"] = model_id

        results["experiments"]["aviation"] = (
            experiment_aviation(model_id)
        )

        results["experiments"]["transformer"] = (
            experiment_transformer(model_id)
        )

        results["experiments"]["code_generation"] = (
            experiment_code(model_id)
        )

        results["experiments"]["streaming"] = (
            experiment_streaming(model_id)
        )

        results["experiments"]["benchmark"] = (
            experiment_benchmark(
                model_id,
                iterations=5
            )
        )

        results["status"] = "COMPLETED"

    except Exception as e:

        results["status"] = "ERROR"

        results["error"] = repr(e)

        print("\n[EXPERIMENT ERROR]")
        print(repr(e))

    finally:

        results["total_experiment_seconds"] = (
            time.perf_counter() - experiment_start
        )

        filename = (
            "furiosa_npu_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".json"
        )

        output_file = OUTPUT_DIR / filename

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                results,
                f,
                ensure_ascii=False,
                indent=4
            )

        print("\n" + "=" * 72)
        print("EXPERIMENT FINISHED")
        print("=" * 72)

        print("Status:", results["status"])

        print(
            "Saved:",
            output_file.resolve()
        )

        print(
            "Total Time:",
            round(
                results["total_experiment_seconds"],
                2
            ),
            "sec"
        )

        if results["status"] == "COMPLETED":

            print("\nAll experiments completed.")

            chat_choice = input(
                "\nStart interactive chat? (y/n): "
            ).strip().lower()

            if chat_choice == "y":

                interactive_chat(results["model"])


# ============================================================
# 13. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
