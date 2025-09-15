import requests
import threading
import time
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
AGENT_SERVER_URL = "http://127.0.0.1:8001/invoke_agent"
MCP_SERVER_URL = "http://127.0.0.1:8002"
NUM_CONCURRENT_USERS = 5
TEST_DURATION_MINUTES = 5
TEST_QUESTION = "What was the status of the last product Jane Doe ordered?"
LATENCY_PROFILES = ["ultra", "super", "standard"]
TOTAL_RUNS = 5 # The number of times to repeat the full 15-minute experiment
OUTPUT_CSV_PATH = "load_test_results.csv"
OUTPUT_CHART_PATH = "latency_impact_on_throughput.png"


# --- Global State for Tracking (reset for each profile) ---
completed_requests = 0
start_time = 0
console = Console()

# --- Thread-Safe Counter ---
lock = threading.Lock()

def increment_counter():
    """Safely increments the global request counter."""
    global completed_requests
    with lock:
        completed_requests += 1

def set_latency_profile(profile: str):
    """Makes an API call to the MCP server to set the latency profile."""
    console.print(f"\n[bold yellow]Setting latency profile to: {profile.upper()}[/bold yellow]")
    try:
        response = requests.post(f"{MCP_SERVER_URL}/set_latency_profile", json={"profile": profile})
        if response.status_code == 200:
            console.print(f"[green]Successfully set latency profile to '{response.json()['new_profile']}'.[/green]")
            return True
        else:
            console.print(f"[bold red]Error setting latency profile. Status: {response.status_code}, Detail: {response.text}[/bold red]")
            return False
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Failed to connect to MCP server to set latency: {e}[/bold red]")
        return False

def worker_thread(progress, task_id, end_time):
    """The function that each simulated user (thread) will run."""
    while time.time() < end_time:
        try:
            response = requests.post(
                AGENT_SERVER_URL,
                json={"question": TEST_QUESTION},
                timeout=120 # Generous timeout for a single request
            )
            if response.status_code == 200:
                increment_counter()
                progress.update(task_id, advance=1, description=f"[cyan]User {task_id+1} completed request")
            else:
                console.print(f"[bold red]User {task_id+1} received error: {response.status_code}[/bold red]")
        except requests.exceptions.RequestException as e:
            console.print(f"[bold red]User {task_id+1} request failed: {e}[/bold red]")
        
        time.sleep(0.1) 
    
    progress.update(task_id, description=f"[green]User {task_id+1} finished")

def run_test_for_profile(profile: str):
    """Sets up and runs the 5-minute load test for a single latency profile."""
    global start_time, completed_requests
    
    if not set_latency_profile(profile):
        return None # Skip test if we can't set the profile

    completed_requests = 0
    start_time = time.time()
    end_time = start_time + (TEST_DURATION_MINUTES * 60)
    
    threads = []
    
    with Progress(console=console) as progress:
        console.print(f"\n[bold]Starting load test for '{profile.upper()}' profile with {NUM_CONCURRENT_USERS} users for {TEST_DURATION_MINUTES} minutes...[/bold]")
        
        user_tasks = [progress.add_task(f"[cyan]User {i+1}", total=None) for i in range(NUM_CONCURRENT_USERS)]

        for i in range(NUM_CONCURRENT_USERS):
            thread = threading.Thread(target=worker_thread, args=(progress, i, end_time))
            threads.append(thread)
            thread.start()
        
        main_task = progress.add_task("[green]Overall Progress", total=TEST_DURATION_MINUTES * 60)
        while time.time() < end_time:
            elapsed = time.time() - start_time
            progress.update(main_task, completed=elapsed)
            time.sleep(1)
        progress.update(main_task, completed=TEST_DURATION_MINUTES * 60)

        console.print("\n[bold yellow]Test duration complete. Waiting for active requests to finish...[/bold yellow]")
        for thread in threads:
            thread.join()

    total_time_seconds = time.time() - start_time
    requests_per_minute = (completed_requests / total_time_seconds) * 60 if total_time_seconds > 0 else 0

    console.print("\n" + "="*50)
    console.print(f"[bold magenta]LOAD TEST COMPLETE for '{profile.upper()}'[/bold magenta]")
    console.print(f"[bold green]Total Completed Requests:[/bold green] {completed_requests}")
    console.print(f"[bold cyan]Average Requests Per Minute:[/bold cyan] {requests_per_minute:.2f}")
    console.print("="*50)
    
    return {
        "profile": profile,
        "completed_requests": completed_requests,
        "rpm": requests_per_minute
    }

def generate_summary_graph(df: pd.DataFrame):
    """Generates and saves a line chart summarizing the results across all runs."""
    console.print("\n--- [Step 2/3] Generating summary graph ---")
    
    # Pivot the data to have run_number as the index and profiles as columns
    pivot_df = df.pivot(index='run_number', columns='profile', values='completed_requests')
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot a line for each latency profile
    for profile in LATENCY_PROFILES:
        ax.plot(pivot_df.index, pivot_df[profile], marker='o', linestyle='-', label=profile.upper())

    ax.set_title('Agent Throughput vs. Tool Latency (5 Runs)', fontsize=16, pad=20)
    ax.set_xlabel('Experiment Run Number', fontsize=12)
    ax.set_ylabel('Total Completed Requests (in 5 mins)', fontsize=12)
    
    # Set the x-axis to be discrete runs (1, 2, 3, 4, 5)
    ax.set_xticks(range(1, TOTAL_RUNS + 1))
    
    # --- DYNAMIC Y-AXIS SCALING ---
    # Find the minimum number of requests across all runs and profiles
    min_requests = df['completed_requests'].min()
    # Set the bottom of the y-axis to be slightly less than the minimum
    # This "zooms in" on the performance difference.
    ax.set_ylim(bottom=max(0, min_requests - 20)) 
    
    ax.legend(title='Latency Profile')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(OUTPUT_CHART_PATH)
    console.print(f"[green]Graph saved successfully to [cyan]{OUTPUT_CHART_PATH}[/cyan][/green]")


def main():
    """Main function to orchestrate the entire multi-run, multi-profile test."""
    all_results = []
    
    for i in range(1, TOTAL_RUNS + 1):
        console.print("\n" + "#"*60)
        console.print(f"[bold cyan]STARTING EXPERIMENT RUN {i} of {TOTAL_RUNS}[/bold cyan]")
        console.print("#"*60 + "\n")
        
        for profile in LATENCY_PROFILES:
            result = run_test_for_profile(profile)
            if result:
                result['run_number'] = i
                all_results.append(result)
            
            if profile != LATENCY_PROFILES[-1]:
                console.print("\n[bold blue]Pausing for 15 seconds before the next test...[/bold blue]")
                time.sleep(15)

    # --- Process and Save Final Results ---
    console.print("\n\n" + "#"*60)
    console.print("[bold cyan]FINAL CONSOLIDATED EXPERIMENT RESULTS[/bold cyan]")
    console.print("#"*60 + "\n")

    results_df = pd.DataFrame(all_results)
    console.print(f"--- [Step 1/3] Saving all raw results to [cyan]{OUTPUT_CSV_PATH}[/cyan] ---")
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    console.print("[green]CSV file saved successfully.[/green]")
    
    # Generate and save the graph
    generate_summary_graph(results_df)

    # Print final summary table
    summary_table = Table(title="Average Agent Throughput vs. Tool Latency", show_header=True, header_style="bold magenta")
    summary_table.add_column("Latency Profile", style="dim", width=20)
    summary_table.add_column("Avg. Requests (5 min)", justify="right")
    summary_table.add_column("Avg. Requests Per Minute", justify="right")

    final_summary = results_df.groupby('profile')[['completed_requests', 'rpm']].mean().reindex(LATENCY_PROFILES)
    for profile, row in final_summary.iterrows():
        summary_table.add_row(
            profile.upper(),
            f"{row['completed_requests']:.2f}",
            f"{row['rpm']:.2f}"
        )
    
    console.print("\n--- [Step 3/3] Final Summary ---")
    console.print(summary_table)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Load test interrupted by user.[/bold red]")

