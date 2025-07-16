import os
import json
import subprocess
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datasets import load_dataset
import tempfile
import time

class CursorSWEBenchTester:
    def __init__(self, workspace_dir: str = "./data/cursor_workspace", results_dir: str = "./data/cursor_results"):
        self.workspace_dir = Path(workspace_dir)
        self.results_dir = Path(results_dir)
        self.workspace_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Load SWE-bench Lite dataset
        print("Loading SWE-bench Lite dataset...")
        self.dataset = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')
        print(f"Loaded {len(self.dataset)} instances")
        
    def setup_repositories(self, start_idx: int = 0, end_idx: Optional[int] = None):
        """Setup repositories for testing"""
        if end_idx is None:
            end_idx = len(self.dataset)
            
        print(f"Setting up repositories for instances {start_idx} to {end_idx}")
        
        subset = self.dataset.select(range(start_idx, end_idx))
        for i, instance in enumerate(subset, start_idx):
            print(f"\n[{i+1}/{len(self.dataset)}] Setting up {instance['instance_id']}")
            
            repo_name = instance['repo'].replace('/', '_')
            instance_dir = self.workspace_dir / f"instance_{i:03d}_{repo_name}"
            
            # Skip if already exists
            if instance_dir.exists():
                print(f"  Repository already exists at {instance_dir}")
                continue
                
            try:
                # Clone repository
                print(f"  Cloning {instance['repo']}...")
                subprocess.run([
                    'git', 'clone', f"https://github.com/{instance['repo']}.git", str(instance_dir)
                ], check=True, capture_output=True)
                
                # Checkout base commit
                print(f"  Checking out commit {instance['base_commit'][:8]}...")
                subprocess.run([
                    'git', '-C', str(instance_dir), 'checkout', instance['base_commit']
                ], check=True, capture_output=True)
                
                # Save instance metadata
                metadata = {
                    'instance_id': instance['instance_id'],
                    'repo': instance['repo'],
                    'base_commit': instance['base_commit'],
                    'problem_statement': instance['problem_statement'],
                    'hints_text': instance.get('hints_text', ''),
                    'test_patch': instance['test_patch'],
                    'patch': instance['patch']
                }
                
                with open(instance_dir / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
                print(f"   Setup complete: {instance_dir}")
                
            except subprocess.CalledProcessError as e:
                print(f"   Error setting up {instance['instance_id']}: {e}")
                continue
            except Exception as e:
                print(f"   Unexpected error: {e}")
                continue
                
    def generate_prompts(self, start_idx: int = 0, end_idx: Optional[int] = None):
        """Generate standardized prompts for Cursor"""
        if end_idx is None:
            end_idx = len(self.dataset)
            
        prompts_file = self.results_dir / 'cursor_prompts.md'
        
        with open(prompts_file, 'w') as f:
            f.write("# Cursor SWE-bench Lite Prompts\n\n")
            f.write("Use these prompts systematically with Cursor for each instance.\n\n")
            f.write("## Instructions\n")
            f.write("1. Open the repository directory in Cursor\n")
            f.write("2. Copy the prompt for the instance\n")
            f.write("3. Ask Cursor to implement the fix\n")
            f.write("4. Save the changes and generate a git diff\n")
            f.write("5. Record the results in the results file\n\n")
            f.write("---\n\n")
            
            subset = self.dataset.select(range(start_idx, end_idx))
            for i, instance in enumerate(subset, start_idx):
                repo_name = instance['repo'].replace('/', '_')
                instance_dir = self.workspace_dir / f"instance_{i:03d}_{repo_name}"
                
                f.write(f"## Instance {i+1}: {instance['instance_id']}\n\n")
                f.write(f"**Repository**: {instance['repo']}\n")
                f.write(f"**Directory**: `{instance_dir}`\n")
                f.write(f"**Base Commit**: {instance['base_commit'][:8]}\n\n")
                
                # Generate the prompt
                prompt = self._create_cursor_prompt(instance)
                f.write("### Prompt for Cursor:\n\n")
                f.write("```\n")
                f.write(prompt)
                f.write("\n```\n\n")
                
                f.write("### Expected Files to Modify:\n")
                # Try to extract files from the patch
                patch_lines = instance['patch'].split('\n')
                modified_files = []
                for line in patch_lines:
                    if line.startswith('--- a/') or line.startswith('+++ b/'):
                        file_path = line.split('/', 1)[1] if '/' in line else line[4:]
                        if file_path not in modified_files and not file_path.startswith('dev/null'):
                            modified_files.append(file_path)
                
                for file_path in modified_files[:5]:  # Limit to first 5 files
                    f.write(f"- `{file_path}`\n")
                    
                f.write(f"\n**Record your results in**: `data/cursor_results/instance_{i:03d}_results.json`\n\n")
                f.write("---\n\n")
                
        print(f"Prompts generated and saved to: {prompts_file}")
        
    def _create_cursor_prompt(self, instance: Dict) -> str:
        """Create a standardized prompt for Cursor"""
        prompt = f"""I need to fix a GitHub issue in the {instance['repo']} repository. 

**Issue Description:**
{instance['problem_statement']}

**Additional Context:**
{instance.get('hints_text', 'No additional hints provided.')}

**Task:**
Please analyze the codebase and implement a fix for this issue. Focus on:
1. Understanding the root cause of the problem
2. Implementing a minimal, targeted fix
3. Ensuring the fix doesn't break existing functionality
4. Following the project's coding style and conventions

**Instructions:**
- Make only the necessary changes to fix the issue
- Avoid modifying test files unless absolutely necessary
- Provide clear comments explaining your changes
- Test your solution if possible

Please implement the fix now."""
        
        return prompt
    
    def create_result_template(self, instance_idx: int):
        """Create a template file for recording results"""
        instance = self.dataset[instance_idx]
        repo_name = instance['repo'].replace('/', '_')
        result_file = self.results_dir / f"instance_{instance_idx:03d}_results.json"
        
        if result_file.exists():
            return result_file
            
        template = {
            "instance_id": instance['instance_id'],
            "instance_idx": instance_idx,
            "repo": instance['repo'],
            "status": "pending",  # pending, completed, failed, skipped
            "cursor_response": "",
            "files_modified": [],
            "git_diff": "",
            "notes": "",
            "time_spent_minutes": 0,
            "difficulty_rating": "",  # easy, medium, hard
            "success_rating": ""  # success, partial, failed
        }
        
        with open(result_file, 'w') as f:
            json.dump(template, f, indent=2)
            
        return result_file
    
    def collect_results(self):
        """Collect and process all results"""
        print("Collecting results...")
        
        predictions = []
        summary = {
            "total_instances": len(self.dataset),
            "completed": 0,
            "pending": 0,
            "failed": 0,
            "skipped": 0
        }
        
        for i in range(len(self.dataset)):
            result_file = self.results_dir / f"instance_{i:03d}_results.json"
            
            if not result_file.exists():
                print(f"Missing results for instance {i:03d}")
                summary["pending"] += 1
                continue
                
            with open(result_file, 'r') as f:
                result = json.load(f)
                
            summary[result["status"]] += 1
            
            if result["status"] == "completed" and result["git_diff"]:
                prediction = {
                    "instance_id": result["instance_id"],
                    "model_name_or_path": "cursor",
                    "model_patch": result["git_diff"]
                }
                predictions.append(prediction)
                
        # Save predictions for SWE-bench evaluation
        predictions_file = self.results_dir / "cursor_predictions.jsonl"
        with open(predictions_file, 'w') as f:
            for prediction in predictions:
                f.write(json.dumps(prediction) + '\n')
            
        # Save summary
        summary_file = self.results_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\nResults Summary:")
        print(f"  Total instances: {summary['total_instances']}")
        print(f"  Completed: {summary['completed']}")
        print(f"  Pending: {summary['pending']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Skipped: {summary['skipped']}")
        print(f"\nPredictions saved to: {predictions_file}")
        print(f"Summary saved to: {summary_file}")
        
        return predictions_file
    
    
def main():
    parser = argparse.ArgumentParser(description="SWE-bench Lite testing script for Cursor")
    parser.add_argument("--mode", choices=["setup", "prompt", "collect", "guide"], 
                       required=True, help="Mode to run")
    parser.add_argument("--start", type=int, default=0, 
                       help="Start index for instances")
    parser.add_argument("--end", type=int, default=None,
                       help="End index for instances")
    parser.add_argument("--workspace", default="./data/cursor_workspace",
                       help="Workspace directory for repositories")
    parser.add_argument("--results", default="./data/cursor_results",
                       help="Results directory")
    
    args = parser.parse_args()
    
    tester = CursorSWEBenchTester(args.workspace, args.results)
    
    if args.mode == "setup":
        tester.setup_repositories(args.start, args.end)
        print("\n Setup complete! Repositories are ready in data/cursor_workspace/")
        print("Next step: Run 'python cursor_swebench_tester.py --mode prompt' to generate prompts")
        
    elif args.mode == "prompt":
        tester.generate_prompts(args.start, args.end)
        # Create result templates
        for i in range(args.start, args.end or len(tester.dataset)):
            tester.create_result_template(i)
        print("\n Prompts generated! Check data/cursor_results/cursor_prompts.md")
        print("Next step: Use Cursor with the prompts and record results")
        
    elif args.mode == "collect":
        predictions_file = tester.collect_results()
        print(f"\n Results collected! Run evaluation with:")
        print(f"python -m swebench.harness.run_evaluation \\")
        print(f"    --dataset_name princeton-nlp/SWE-bench_Lite \\")
        print(f"    --predictions_path {predictions_file} \\")
        print(f"    --max_workers 4 \\")
        print(f"    --run_id cursor_evaluation \\")
        print(f"If on MacOS, add the following flag:")
        print(f"    --namespace ''")
        

if __name__ == "__main__":
    main() 