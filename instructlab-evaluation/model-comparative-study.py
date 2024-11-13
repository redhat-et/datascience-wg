import pandas as pd
import json
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

class ModelComparison:
    def __init__(self):
        # Load both models' results and evaluation data
        print("Loading model data...")
        self.granite_report = json.load(open('granite/evaluation/final_report.json'))
        self.merlinite_report = json.load(open('merlinite/evaluation/final_report.json'))
        
        # Load scores data
        self.granite_bert = pd.read_json('granite/evaluation/bert_scores.json')
        self.merlinite_bert = pd.read_json('merlinite/evaluation/bert_scores.json')
        
        # Load full results and merge with scores
        self.granite_data = pd.read_json('granite/results/final_responses_20241109_210419.json')
        self.merlinite_data = pd.read_json('merlinite/results/fixed_responses.json')
        
        # Add scores to main dataframes
        self.granite_data['bert_f1'] = self.granite_bert['bert_f1']
        self.merlinite_data['bert_f1'] = self.merlinite_bert['bert_f1']

    def get_overall_comparison(self):
        """Compare overall metrics between models"""
        metrics = {
            'Metric': [
                'Total Responses',
                'Number of Categories',
                'Average ROUGE-1 Score',
                'Average BERT F1 Score',
                'Average Response Length',
                'Average Reference Length'
            ],
            'Granite': [
                self.granite_report['summary']['total_responses'],
                self.granite_report['summary']['number_of_categories'],
                self.granite_report['summary']['average_rouge1_score'],
                self.granite_report['summary']['average_bert_f1_score'],
                self.granite_report['summary']['average_response_length'],
                self.granite_report['summary']['average_reference_length']
            ],
            'Merlinite': [
                self.merlinite_report['summary']['total_responses'],
                self.merlinite_report['summary']['number_of_categories'],
                self.merlinite_report['summary']['average_rouge1_score'],
                self.merlinite_report['summary']['average_bert_f1_score'],
                self.merlinite_report['summary']['average_response_length'],
                self.merlinite_report['summary']['average_reference_length']
            ]
        }
        df = pd.DataFrame(metrics)
        # Format numerical columns
        for col in ['Granite', 'Merlinite']:
            df[col] = pd.to_numeric(df[col], errors='ignore').round(4)
        return df

    def find_best_worst_examples(self, model_data, scores_type='bert_f1'):
        """Find best and worst examples based on scores"""
        best_idx = model_data[scores_type].idxmax()
        worst_idx = model_data[scores_type].idxmin()
        
        best_example = {
            'category': model_data.loc[best_idx, 'category'],
            'question': model_data.loc[best_idx, 'question'],
            'reference': model_data.loc[best_idx, 'reference_answer'],
            'model_response': model_data.loc[best_idx, 'model_response'],
            'score': model_data.loc[best_idx, scores_type]
        }
        
        worst_example = {
            'category': model_data.loc[worst_idx, 'category'],
            'question': model_data.loc[worst_idx, 'question'],
            'reference': model_data.loc[worst_idx, 'reference_answer'],
            'model_response': model_data.loc[worst_idx, 'model_response'],
            'score': model_data.loc[worst_idx, scores_type]
        }
        
        return best_example, worst_example

    def category_comparison(self):
        """Compare performance across categories"""
        granite_cats = pd.DataFrame(self.granite_report['category_performance']).T
        merlinite_cats = pd.DataFrame(self.merlinite_report['category_performance']).T
        
        comparison = pd.DataFrame({
            'Granite_ROUGE1': granite_cats['avg_rouge1'],
            'Granite_BERT': granite_cats['avg_bert_f1'],
            'Merlinite_ROUGE1': merlinite_cats['avg_rouge1'],
            'Merlinite_BERT': merlinite_cats['avg_bert_f1']
        }).round(4)
        
        return comparison

    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        print("\n=== Model Comparison Report ===\n")
        
        # Overall Metrics
        print("1. Overall Performance Comparison:")
        print(tabulate(self.get_overall_comparison(), headers='keys', tablefmt='pretty', showindex=False))
        print("\n")

        # Best/Worst Examples for both models
        print("2. Best and Worst Examples:\n")
        for model_name, model_data in [('Granite', self.granite_data), ('Merlinite', self.merlinite_data)]:
            best, worst = self.find_best_worst_examples(model_data)
            
            print(f"\n{model_name} Best Example (BERT F1 Score: {best['score']:.4f}):")
            print(f"Category: {best['category']}")
            print(f"Q: {best['question']}")
            print(f"A (Model): {best['model_response'][:300]}...")
            print(f"A (Reference): {best['reference'][:300]}...")
            
            print(f"\n{model_name} Worst Example (BERT F1 Score: {worst['score']:.4f}):")
            print(f"Category: {worst['category']}")
            print(f"Q: {worst['question']}")
            print(f"A (Model): {worst['model_response'][:300]}...")
            print(f"A (Reference): {worst['reference'][:300]}...")
            print("\n" + "-"*80 + "\n")

        # Category Analysis
        cat_comparison = self.category_comparison()
        print("\n3. Category Performance Summary:")
        print(f"Total Categories: {len(cat_comparison)}")
        
        # Find categories where each model excels
        granite_better = cat_comparison[
            cat_comparison['Granite_BERT'] > cat_comparison['Merlinite_BERT']
        ].index.tolist()
        
        merlinite_better = cat_comparison[
            cat_comparison['Merlinite_BERT'] > cat_comparison['Granite_BERT']
        ].index.tolist()
        
        print(f"\nGranite performs better in {len(granite_better)} categories")
        print(f"Merlinite performs better in {len(merlinite_better)} categories")
        
        # Top 5 categories for each model
        print("\nTop 5 Categories for Granite:")
        top_granite = cat_comparison.sort_values('Granite_BERT', ascending=False).head()
        print(tabulate(top_granite[['Granite_BERT', 'Merlinite_BERT']], headers='keys', tablefmt='pretty'))
        
        print("\nTop 5 Categories for Merlinite:")
        top_merlinite = cat_comparison.sort_values('Merlinite_BERT', ascending=False).head()
        print(tabulate(top_merlinite[['Granite_BERT', 'Merlinite_BERT']], headers='keys', tablefmt='pretty'))

        # Save report to file
        with open('model_comparison_report.txt', 'w') as f:
            f.write("=== Model Comparison Report ===\n\n")
            f.write("This report compares the performance of Granite and Merlinite models across multiple dimensions.\n\n")
            f.write("Key Findings:\n")
            f.write(f"1. Granite excels in {len(granite_better)} categories\n")
            f.write(f"2. Merlinite excels in {len(merlinite_better)} categories\n")
            f.write(f"3. Average BERT F1 Scores:\n")
            f.write(f"   - Granite: {self.granite_report['summary']['average_bert_f1_score']:.4f}\n")
            f.write(f"   - Merlinite: {self.merlinite_report['summary']['average_bert_f1_score']:.4f}\n\n")
            
            # Add detailed category analysis
            f.write("\nDetailed Category Analysis:\n")
            f.write(cat_comparison.to_string())

def main():
    comparison = ModelComparison()
    comparison.generate_comparison_report()

if __name__ == "__main__":
    main()
