import time
import json
import os
import asyncio
import aiohttp
import argparse # For command-line arguments
from collections import defaultdict
from datetime import datetime
from itertools import cycle
from tqdm.asyncio import tqdm_asyncio

# --- Configuration (Default values) ---
# These can be overridden by command-line arguments.
# Default port for the API server
DEFAULT_PORT = 8001

# Define the folder to save the results.
OUTPUT_FOLDER = "benchmark_output"

# Define the base name for the output file if no command-line argument is given.
OUTPUT_FILENAME_BASE = "my_llm_benchmark"

# Number of concurrent requests to send at any given time.
MAX_CONCURRENT_REQUESTS = 50

# Define the number of requests to send for the benchmark PER CATEGORY.
NUM_REQUESTS_PER_CATEGORY = 200

# Maximum number of tokens to generate for each response.
MAX_TOKENS = 150

# Default model name - can be overridden via --model argument
DEFAULT_MODEL_NAME = "openai/gpt-oss-120b"

# --- Prompts ---
CATEGORIZED_PROMPTS = {
    "information_retrieval": [
        "What is the capital of France?",
        "What is the longest river in the world?",
        "Who was the first person to walk on the moon?",
        "What is the chemical formula for water?",
        "In what year did World War II end?",
        "What is the population of Tokyo?",
        "Who wrote 'To Kill a Mockingbird'?",
        "What is the speed of light in a vacuum?",
        "How many bones are in the adult human body?",
        "What is the boiling point of water at sea level in Celsius?",
        "What is the largest ocean on Earth?",
        "Who painted the Mona Lisa?",
        "What is the chemical symbol for gold?",
        "What is the main function of the heart?",
        "What is the name of the fourth planet from the sun?",
        "Who invented the telephone?",
        "What is the capital of Brazil?",
        "What is the square root of 144?",
        "What is the smallest country in the world by area?",
        "What is the main ingredient in guacamole?",
        "What is the deepest point in the ocean?",
        "Who discovered penicillin?",
        "What is the capital of Canada?",
        "What is the largest land animal?",
        "What is the process of photosynthesis?",
        "What is the chemical symbol for oxygen?",
        "What is the freezing point of water in Celsius?",
        "What is the theory of evolution?",
        "Who was Albert Einstein?",
        "What is a supernova?",
        "What is the Earth's atmosphere made of?",
        "What is the capital of Russia?",
        "Who was Leonardo da Vinci?",
        "What is the main purpose of the lungs?",
        "What is a galaxy?",
        "What is the capital of India?",
        "Who was Stephen Hawking?",
        "What is a black hole?",
        "What is the capital of Egypt?",
        "Who was Nikola Tesla?",
        "What is the main function of the stomach?",
        "What is the process of cell division?",
        "What is the capital of Germany?",
        "Who was Alan Turing?",
        "What is the main function of the liver?",
        "What is the capital of Italy?",
        "Who was Johann Sebastian Bach?",
        "What is the main function of the pancreas?",
        "What is the capital of Spain?",
        "Who was Ludwig van Beethoven?",
    ],
    "reasoning": [
        "Explain the relationship between supply and demand.",
        "Describe the pros and cons of nuclear energy.",
        "How can you prove the Pythagorean theorem?",
        "If all humans are mortal and I am a human, what can you conclude?",
        "What is the difference between weather and climate?",
        "Explain why a ship floats but a rock sinks.",
        "Given a list of numbers, how would you find the median?",
        "Explain the domino effect in a political context.",
        "How would you sort a list of names alphabetically?",
        "Explain the concept of a balanced diet using a metaphor.",
        "If you have a 3-gallon jug and a 5-gallon jug, how do you measure out exactly 4 gallons of water?",
        "What logical steps would you take to debug a simple computer program?",
        "Explain why the seasons change.",
        "Why is it important to recycle?",
        "Explain the process of creating a new law in the United States.",
        "How does a computer processor execute instructions?",
        "What are the arguments for and against raising the minimum wage?",
        "Explain the concept of compound interest.",
        "How would you explain the value of education to a young child?",
        "Describe a strategy for winning a game of chess.",
        "What are the ethical implications of artificial intelligence?",
        "Explain the steps to solve a Rubik's Cube.",
        "Why do birds migrate?",
        "Explain how the internet works in simple terms.",
        "Describe the process of a bill becoming a law.",
        "How would you plan a road trip across the country?",
        "Explain the theory of relativity using a simple analogy.",
        "Why is it important to have a budget?",
        "What are the logical fallacies in the statement, 'Everyone is doing it, so it must be right'?",
        "Describe how to make a perfect cup of coffee.",
        "Explain the difference between a virus and a bacterium.",
        "How does a cell's nucleus function?",
        "What is the reasoning behind daylight saving time?",
        "Explain how a steam engine works.",
        "What are the benefits of a diverse team?",
        "Describe the steps to build a simple wooden chair.",
        "Why do we have a leap year?",
        "Explain the concept of a food chain.",
        "How would you design a personal fitness plan?",
        "Explain the difference between a polygon and a circle.",
        "What is the logic behind a negotiation strategy?",
        "Describe the steps to set up a small business.",
        "How does the Earth's rotation affect our daily lives?",
        "Explain the concept of gravity.",
        "How would you prepare for a natural disaster?",
        "What is the rationale for traffic laws?",
        "Explain the difference between a hypothesis and a theory.",
        "How would you convince someone to adopt a pet?",
        "What is the logic behind an algorithm?",
        "Explain why it's a good idea to back up your data.",
    ],
    "creative_text_generation": [
        "Write a short poem about a forgotten toy.",
        "Describe a perfect day in your own words.",
        "Write a short story about a talking animal.",
        "What is the most beautiful color and why?",
        "Write a haiku about a rainy day.",
        "Describe the feeling of joy without using the word 'happy'.",
        "What would you do if you could fly?",
        "Write a dialogue between a curious robot and a wise tree.",
        "What is the sound of silence?",
        "Write a poem about the ocean.",
        "Describe the taste of a summer breeze.",
        "Imagine a world without music.",
        "What does freedom mean to you?",
        "Write a short story about a message in a bottle.",
        "If you could travel anywhere in time, where would you go?",
        "Describe your favorite memory.",
        "Write a poem about the color blue.",
        "What is the most important invention of all time?",
        "Describe the perfect pizza.",
        "Write a journal entry from the perspective of a dog.",
        "What is the meaning of life?",
        "Write a short story about a time traveler who loses their way.",
        "Describe a magical forest.",
        "What is a dream?",
        "Write a poem about the first day of spring.",
        "What is the most powerful emotion?",
        "Describe the feeling of nostalgia.",
        "What is the most challenging thing you've ever done?",
        "Write a short story about a forgotten toy.",
        "Describe the taste of fresh-baked bread.",
        "What is a lie?",
        "Write a poem about the moon.",
        "What is a good way to relax?",
        "Describe a day in the life of a cloud.",
        "What is the biggest mystery of the universe?",
        "Write a short story about a brave knight and a timid dragon.",
        "What is the meaning of a smile?",
        "Write a poem about the sound of rain.",
        "What is a fun way to learn something new?",
        "Describe a new kind of animal.",
        "What is courage?",
        "Write a short story about a lonely alien.",
        "What is a bad habit you'd like to break?",
        "Write a poem about the first snow of the year.",
        "What is the best piece of advice you've ever received?",
        "Describe a city in the future.",
        "What is a good joke?",
        "Write a short story about a magical key.",
        "What is the best way to make a friend?",
        "Write a poem about a firefly.",
    ],
    "task_completion": [
        "Generate a Python function to check if a number is prime.",
        "Write an HTML form for a user login.",
        "Create a CSS snippet to make a button hover effect.",
        "Write a short, professional email to request a meeting.",
        "Generate a grocery list for a week of meals.",
        "Write a 5-step plan to organize a closet.",
        "Create a recipe for a simple chocolate cake.",
        "Write a short guide on how to change a tire.",
        "Generate a list of 10 common houseplants that are easy to care for.",
        "Create a workout plan for a beginner, focusing on core strength.",
        "Write a thank-you note for a job interview.",
        "Generate a simple budget template in a list format.",
        "Create a short speech for a best friend's wedding.",
        "Write a short script for a commercial selling a new coffee machine.",
        "Generate a list of 5 things to do to prepare for a job interview.",
        "Create a list of 10 essential items for a hiking trip.",
        "Write a short summary of a research paper on climate change.",
        "Generate a simple to-do list for a productive workday.",
        "Create a checklist for moving into a new apartment.",
        "Write a short guide on how to plant a tomato plant.",
        "Generate a list of 5 popular tourist attractions in Paris.",
        "Create a simple meal plan for a week.",
        "Write a short guide on how to save money.",
        "Generate a list of 10 classic books everyone should read.",
        "Create a short guide on how to bake bread.",
        "Write a simple resume template.",
        "Generate a list of 5 tips for public speaking.",
        "Create a short guide on how to start a podcast.",
        "Write a short guide on how to write a good blog post.",
        "Generate a list of 5 ways to reduce your carbon footprint.",
        "Create a short guide on how to meditate.",
        "Write a simple cover letter template.",
        "Generate a list of 5 tips for learning a new language.",
        "Create a short guide on how to write a good email.",
        "Write a simple itinerary for a weekend trip.",
        "Generate a list of 5 tips for time management.",
        "Create a short guide on how to create a good presentation.",
        "Write a simple plan for a birthday party.",
        "Generate a list of 5 tips for staying motivated.",
        "Create a short guide on how to be a good leader.",
        "Write a simple plan for a hiking trip.",
        "Generate a list of 5 tips for a healthy diet.",
        "Create a short guide on how to be a good friend.",
        "Write a simple plan for a study session.",
        "Generate a list of 5 tips for stress management.",
        "Create a short guide on how to write a good story.",
        "Write a simple plan for a workout routine.",
        "Generate a list of 5 tips for giving a good speech.",
        "Create a short guide on how to be a good listener.",
        "Write a simple plan for a garden.",
    ]
}


async def send_request(session, semaphore, prompt, category, request_num, max_tokens, model_name, api_url):
    """
    Sends a single asynchronous request to the API and captures metrics.
    A semaphore is used to limit the number of concurrent requests.
    """
    async with semaphore:
        request_payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True
        }

        start_total_time = time.perf_counter()

        try:
            async with session.post(api_url, json=request_payload) as response:
                response.raise_for_status()

                first_token_time = None
                tokens_count = 0
                full_response_content = ""

                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        json_data = line[len("data: "):]
                        if json_data.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(json_data)
                            content = data['choices'][0].get('delta', {}).get('content', '')
                            if content:
                                if first_token_time is None:
                                    first_token_time = time.perf_counter() - start_total_time
                                full_response_content += content
                                tokens_count += 1
                        except (json.JSONDecodeError, KeyError):
                            continue

                end_total_time = time.perf_counter()
                total_time = end_total_time - start_total_time
                tps = tokens_count / total_time if total_time > 0 and tokens_count > 0 else 0

                return {
                    "status": "success",
                    "time_to_first_token": first_token_time if first_token_time is not None else total_time,
                    "tokens_per_second": tps,
                }
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            return {
                "status": "error",
                "error": str(e),
            }


def write_results_to_file(summary_text, run_config):
    """Writes the final summary to a file."""
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    # Determine the final output path
    if run_config['output_file']:
        # Use the filename provided via command-line argument
        final_output_path = os.path.join(OUTPUT_FOLDER, run_config['output_file'])
    else:
        # Fallback to the default naming scheme with a timestamp
        base_name = OUTPUT_FILENAME_BASE
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_{timestamp}.txt")

    with open(final_output_path, "w") as f:
        f.write(summary_text)

    print(f"Results saved to '{final_output_path}'.")

def process_and_display_results(results, model_name):
    """
    Calculates final averages, prints them to the console,
    and returns the summary string for file writing.
    """
    successful_results = [r for r in results if r['status'] == 'success']

    if not successful_results:
        print("\nBenchmark completed with no successful requests.")
        return None

    all_ttft = [r['time_to_first_token'] for r in successful_results]
    all_tps = [r['tokens_per_second'] for r in successful_results]

    avg_ttft = sum(all_ttft) / len(all_ttft)
    avg_tps = sum(all_tps) / len(all_tps)
    
    # Create the summary string
    summary_lines = [
        f"Model: {model_name}",
        f"Average Time to First Token: {avg_ttft:.4f}s",
        f"Average Tokens per Second: {avg_tps:.2f}"
    ]
    summary_text = "\n".join(summary_lines)

    # Print to console
    print("\n--- Benchmark Complete ---")
    print(summary_text)
    
    return summary_text


async def run_benchmark(cli_args):
    """Main function to set up and run the concurrent benchmark."""
    # Determine the final values, overriding defaults if CLI args are provided.
    final_port = cli_args.port if cli_args.port is not None else DEFAULT_PORT
    final_concurrent_requests = cli_args.concurrent_requests if cli_args.concurrent_requests is not None else MAX_CONCURRENT_REQUESTS
    final_num_requests = cli_args.num_requests if cli_args.num_requests is not None else NUM_REQUESTS_PER_CATEGORY
    final_max_tokens = cli_args.max_tokens if cli_args.max_tokens is not None else MAX_TOKENS
    final_model_name = cli_args.model if cli_args.model is not None else DEFAULT_MODEL_NAME
    
    # Construct API URL from port
    api_url = f"http://localhost:{final_port}/v1/chat/completions"
    
    run_config = {
        'output_file': cli_args.output_file,
    }

    total_requests = final_num_requests * len(CATEGORIZED_PROMPTS)
    print("--- Starting Benchmark ---")
    print(f"Model: {final_model_name}")
    print(f"API URL: {api_url}")
    print(f"Total Requests: {total_requests}")
    print(f"Concurrent Requests: {final_concurrent_requests}\n")

    semaphore = asyncio.Semaphore(final_concurrent_requests)
    connector = aiohttp.TCPConnector(limit=final_concurrent_requests)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for category, prompts in CATEGORIZED_PROMPTS.items():
            prompt_iterator = cycle(prompts)
            for i in range(final_num_requests):
                prompt = next(prompt_iterator)
                task = send_request(session, semaphore, prompt, category, i + 1, final_max_tokens, final_model_name, api_url)
                tasks.append(task)
        
        results = await tqdm_asyncio.gather(*tasks, desc="Running benchmark")

    summary_text = process_and_display_results(results, final_model_name)
    if summary_text:
        write_results_to_file(summary_text, run_config)

if __name__ == "__main__":
    # To run this script, you first need to install the required libraries:
    # pip install aiohttp tqdm

    # Setup command-line argument parser
    parser = argparse.ArgumentParser(description="Run a concurrent benchmark test on an LLM API.")
    parser.add_argument("-p", "--port", type=int,
                        help=f"The port number for the API server (default: {DEFAULT_PORT})")
    parser.add_argument("-m", "--model", type=str,
                        help=f"The model name to use for the benchmark (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("-o", "--output_file", type=str,
                        help="Specify the output filename.")
    parser.add_argument("-c", "--concurrent_requests", type=int,
                        help=f"The maximum number of concurrent requests (default: {MAX_CONCURRENT_REQUESTS}).")
    parser.add_argument("-n", "--num_requests", type=int,
                        help=f"The number of requests to send per category (default: {NUM_REQUESTS_PER_CATEGORY}).")
    parser.add_argument("-t", "--max_tokens", type=int,
                        help=f"The maximum number of tokens to generate per response (default: {MAX_TOKENS}).")
    
    args = parser.parse_args()

    asyncio.run(run_benchmark(cli_args=args))