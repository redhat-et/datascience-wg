import json
import subprocess
import argparse
from pathlib import Path
import sys

def capture_git_diff(instance_dir: Path) -> str:
    """Capture git diff from the repository"""
    try:
        result = subprocess.run([
            'git', '-C', str(instance_dir), 'diff'
        ], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error capturing git diff: {e}")
        return ""

def get_modified_files(instance_dir: Path) -> list:
    """Get list of modified files"""
    try:
        result = subprocess.run([
            'git', '-C', str(instance_dir), 'diff', '--name-only'
        ], capture_output=True, text=True, check=True)
        return [f.strip() for f in result.stdout.split('\n') if f.strip()]
    except subprocess.CalledProcessError as e:
        print(f"Error getting modified files: {e}")
        return []

def update_result_file(result_file: Path, updates: dict):
    """Update the result JSON file with new data"""
    if not result_file.exists():
        print(f"Result file not found: {result_file}")
        return False
        
    with open(result_file, 'r') as f:
        result_data = json.load(f)
    
    # Update with new data
    result_data.update(updates)
    
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Capture Cursor results for SWE-bench testing")
    parser.add_argument("--instance", type=int, required=True,
                       help="Instance number (0-based index)")
    parser.add_argument("--status", choices=["completed", "failed", "skipped", "pending"],
                       required=True, help="Status of the instance")
    parser.add_argument("--notes", type=str, default="",
                       help="Notes about the solution or issues encountered")
    parser.add_argument("--time", type=int, default=0,
                       help="Time spent in minutes")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="",
                       help="Difficulty rating")
    parser.add_argument("--success", choices=["success", "partial", "failed"], default="",
                       help="Success rating")
    parser.add_argument("--workspace", default="./data/cursor_workspace",
                       help="Workspace directory")
    parser.add_argument("--results", default="./data/cursor_results",
                       help="Results directory")
    parser.add_argument("--no-diff", action="store_true",
                       help="Don't capture git diff (for failed/skipped instances)")
    
    args = parser.parse_args()
    
    # Set up paths
    workspace_dir = Path(args.workspace)
    results_dir = Path(args.results)
    
    # Find instance directory
    instance_dirs = list(workspace_dir.glob(f"instance_{args.instance:03d}_*"))
    if not instance_dirs:
        print(f"No instance directory found for instance {args.instance}")
        print(f"Expected pattern: instance_{args.instance:03d}_*")
        return 1
    
    instance_dir = instance_dirs[0]
    result_file = results_dir / f"instance_{args.instance:03d}_results.json"
    
    print(f"Processing instance {args.instance}")
    print(f"Instance directory: {instance_dir}")
    print(f"Result file: {result_file}")
    
    # Prepare updates
    updates = {
        "status": args.status,
        "notes": args.notes,
        "time_spent_minutes": args.time,
        "difficulty_rating": args.difficulty,
        "success_rating": args.success
    }
    
    # Capture git diff and modified files if status is completed
    if args.status == "completed" and not args.no_diff:
        print("Capturing git diff...")
        git_diff = capture_git_diff(instance_dir)
        modified_files = get_modified_files(instance_dir)
        
        if git_diff:
            updates["git_diff"] = git_diff
            updates["files_modified"] = modified_files
            print(f"Captured diff with {len(git_diff)} characters")
            print(f"Modified files: {modified_files}")
        else:
            print("Warning: No git diff captured. Make sure you've made changes to the repository.")
            choice = input("Continue anyway? (y/n): ")
            if choice.lower() != 'y':
                return 1
    
    # Update result file
    if update_result_file(result_file, updates):
        print(f" Result file updated successfully")
        
        # Show summary
        with open(result_file, 'r') as f:
            result_data = json.load(f)
        
        print(f"\nSummary:")
        print(f"  Instance ID: {result_data['instance_id']}")
        print(f"  Repository: {result_data['repo']}")
        print(f"  Status: {result_data['status']}")
        print(f"  Files modified: {len(result_data.get('files_modified', []))}")
        print(f"  Diff length: {len(result_data.get('git_diff', ''))} characters")
        if result_data.get('notes'):
            print(f"  Notes: {result_data['notes']}")
    else:
        print(" Failed to update result file")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 