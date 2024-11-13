import torch
import pandas as pd
import numpy as np
from bert_score import score
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
from tqdm import tqdm
import os
from datetime import datetime
import gc

class ResponseEvaluator:
    def __init__(self, results_file="results/fixed_responses.json"):
        """Initialize evaluator with results file."""
        print("Loading results...")
        try:
            # First try reading as lines
            self.df = pd.read_json(results_file, lines=True)
        except:
            # If that fails, try regular format
            self.df = pd.read_json(results_file)
            
        self.categories = self.df['category'].unique()
        print(f"Loaded {len(self.df)} responses across {len(self.categories)} categories")
        
        # Create output directory
        os.makedirs('evaluation', exist_ok=True)
        
    def basic_statistics(self):
        """Calculate basic statistics about responses."""
        print("\nCalculating basic statistics...")
        stats = {
            'total_responses': len(self.df),
            'categories': len(self.categories),
            'avg_response_length': self.df['model_response'].str.len().mean(),
            'avg_reference_length': self.df['reference_answer'].str.len().mean(),
            'category_distribution': self.df['category'].value_counts().to_dict()
        }
        
        # Save stats
        with open('evaluation/basic_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def calculate_bert_scores(self):
        """Calculate BERT scores for each response."""
        print("\nCalculating BERT scores...")
        
        # Split into manageable batches to avoid memory issues
        batch_size = 32
        all_refs = self.df['reference_answer'].tolist()
        all_preds = self.df['model_response'].tolist()
        
        P_scores = []
        R_scores = []
        F1_scores = []
        
        for i in tqdm(range(0, len(all_refs), batch_size)):
            batch_refs = all_refs[i:i+batch_size]
            batch_preds = all_preds[i:i+batch_size]
            
            P, R, F1 = score(
                batch_preds, 
                batch_refs,
                lang='en',
                verbose=False,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            P_scores.extend(P.tolist())
            R_scores.extend(R.tolist())
            F1_scores.extend(F1.tolist())
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Add scores to dataframe
        self.df['bert_precision'] = P_scores
        self.df['bert_recall'] = R_scores
        self.df['bert_f1'] = F1_scores
        
        # Save scores
        scores_df = self.df[['category', 'bert_precision', 'bert_recall', 'bert_f1']]
        scores_df.to_json('evaluation/bert_scores.json', orient='records')
        
        print("✓ BERT scores calculated and saved")
    
    def calculate_rouge_scores(self):
        """Calculate ROUGE scores for each response."""
        print("\nCalculating ROUGE scores...")
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        scores = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            score = scorer.score(row['reference_answer'], row['model_response'])
            scores.append({
                'rouge1': score['rouge1'].fmeasure,
                'rouge2': score['rouge2'].fmeasure,
                'rougeL': score['rougeL'].fmeasure
            })
        
        # Add scores to dataframe
        self.df['rouge1'] = [s['rouge1'] for s in scores]
        self.df['rouge2'] = [s['rouge2'] for s in scores]
        self.df['rougeL'] = [s['rougeL'] for s in scores]
        
        # Save scores
        scores_df = self.df[['category', 'rouge1', 'rouge2', 'rougeL']]
        scores_df.to_json('evaluation/rouge_scores.json', orient='records')
        
        print("✓ ROUGE scores calculated and saved")
    
    def analyze_by_category(self):
        """Analyze performance by category."""
        print("\nAnalyzing performance by category...")
        category_stats = defaultdict(dict)
        
        for category in self.categories:
            cat_df = self.df[self.df['category'] == category]
            category_stats[category] = {
                'count': len(cat_df),
                'avg_rouge1': cat_df['rouge1'].mean(),
                'avg_rouge2': cat_df['rouge2'].mean(),
                'avg_rougeL': cat_df['rougeL'].mean(),
                'avg_bert_f1': cat_df['bert_f1'].mean(),
                'avg_response_length': cat_df['model_response'].str.len().mean(),
                'std_rouge1': cat_df['rouge1'].std(),
                'std_bert_f1': cat_df['bert_f1'].std()
            }
        
        # Save category analysis
        with open('evaluation/category_analysis.json', 'w') as f:
            json.dump(category_stats, f, indent=2)
        
        return category_stats
    
    def generate_visualizations(self):
        """Generate various visualizations."""
        print("\nGenerating visualizations...")
        
        # 1. Performance Heatmap (Combined BERT and ROUGE)
        plt.figure(figsize=(15, 10))
        category_scores = pd.DataFrame([{
            'category': cat,
            'ROUGE-1': stats['avg_rouge1'],
            'BERT-F1': stats['avg_bert_f1'],
            'ROUGE-L': stats['avg_rougeL']
        } for cat, stats in self.analyze_by_category().items()])
        
        score_matrix = category_scores.set_index('category')[['ROUGE-1', 'BERT-F1', 'ROUGE-L']]
        sns.heatmap(score_matrix, annot=True, cmap='YlOrRd', center=0.5)
        plt.title('Performance by Category (ROUGE and BERT Scores)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('evaluation/category_heatmap.png')
        plt.close()
        
        # 2. Score Distributions
        plt.figure(figsize=(15, 5))
        metrics_df = self.df[['rouge1', 'rouge2', 'rougeL', 'bert_f1']].melt(
            var_name='Metric', 
            value_name='Score'
        )
        sns.boxplot(data=metrics_df, x='Metric', y='Score')
        plt.title('Distribution of Scores')
        plt.tight_layout()
        plt.savefig('evaluation/score_distributions.png')
        plt.close()
        
        # 3. Length Analysis
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['reference_answer'].str.len(), 
                   self.df['model_response'].str.len(), 
                   alpha=0.5)
        plt.xlabel('Reference Answer Length')
        plt.ylabel('Model Response Length')
        plt.title('Response Length Comparison')
        plt.tight_layout()
        plt.savefig('evaluation/length_comparison.png')
        plt.close()
        
        # 4. Category-wise Performance Comparison
        plt.figure(figsize=(15, 8))
        data = pd.melt(self.df, 
                      id_vars=['category'],
                      value_vars=['rouge1', 'bert_f1'],
                      var_name='metric',
                      value_name='score')
        sns.boxplot(data=data, x='category', y='score', hue='metric')
        plt.xticks(rotation=90)
        plt.title('ROUGE-1 and BERT-F1 Scores by Category')
        plt.tight_layout()
        plt.savefig('evaluation/category_scores.png')
        plt.close()
        
        print("✓ Visualizations generated")
    
    def generate_report(self):
        """Generate a comprehensive evaluation report."""
        print("\nGenerating final report...")
        
        # Calculate all metrics
        basic_stats = self.basic_statistics()
        self.calculate_rouge_scores()
        self.calculate_bert_scores()
        category_stats = self.analyze_by_category()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Get best and worst categories
        category_scores = {cat: stats['avg_bert_f1'] 
                         for cat, stats in category_stats.items()}
        best_category = max(category_scores.items(), key=lambda x: x[1])[0]
        worst_category = min(category_scores.items(), key=lambda x: x[1])[0]
        
        # Generate report
        report = {
            'summary': {
                'total_responses': basic_stats['total_responses'],
                'number_of_categories': basic_stats['categories'],
                'best_performing_category': best_category,
                'worst_performing_category': worst_category,
                'average_rouge1_score': float(self.df['rouge1'].mean()),
                'average_bert_f1_score': float(self.df['bert_f1'].mean()),
                'average_response_length': basic_stats['avg_response_length'],
                'average_reference_length': basic_stats['avg_reference_length']
            },
            'category_performance': category_stats,
            'basic_statistics': basic_stats
        }
        
        # Save report
        with open('evaluation/final_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n=== Evaluation Summary ===")
        print(f"Total Responses: {report['summary']['total_responses']}")
        print(f"Number of Categories: {report['summary']['number_of_categories']}")
        print(f"Best Category: {report['summary']['best_performing_category']}")
        print(f"Worst Category: {report['summary']['worst_performing_category']}")
        print(f"Average ROUGE-1 Score: {report['summary']['average_rouge1_score']:.3f}")
        print(f"Average BERT F1 Score: {report['summary']['average_bert_f1_score']:.3f}")
        
        return report

def main():
    print("=== Starting Evaluation ===")
    evaluator = ResponseEvaluator()
    report = evaluator.generate_report()
    print("\n=== Evaluation Complete ===")
    print("Check the 'evaluation' directory for detailed results and visualizations")

if __name__ == "__main__":
    main()
