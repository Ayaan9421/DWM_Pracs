import itertools
import pandas as pd

class Apriori:
    def __init__(self, min_support=0.5, min_confidence=0.7):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = {}
        self.rules = []

    def fit(self, transactions):
        """
        transactions: list of lists, each sublist represents a transaction
        """
        itemset = set(itertools.chain.from_iterable(transactions))
        itemset = [{item} for item in itemset]
        
        # Generate frequent itemsets
        k = 1
        current_frequent = self._get_frequent_itemsets(itemset, transactions)
        
        while current_frequent:
            self.frequent_itemsets[k] = current_frequent
            k += 1
            
            candidates = self._generate_candidates(current_frequent)
            current_frequent = self._get_frequent_itemsets(candidates, transactions)
        
        # Generate association rules
        self._generate_rules()
        return self

    def _get_frequent_itemsets(self, candidates, transactions):
        """Compute support for each candidate and keep those >= min_support."""
        item_count = {}
        for transaction in transactions:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    item_count[frozenset(candidate)] = item_count.get(frozenset(candidate), 0) + 1
        
        total = len(transactions)
        frequent = {}
        for itemset, count in item_count.items():
            support = count / total
            if support >= self.min_support:
                frequent[itemset] = support
        return frequent

    def _generate_candidates(self, prev_frequent):
        """Join step: combine itemsets to generate candidates of size k+1."""
        prev_frequent_list = list(prev_frequent.keys())
        next_candidates = []
        
        for i in range(len(prev_frequent_list)):
            for j in range(i + 1, len(prev_frequent_list)):
                union = prev_frequent_list[i].union(prev_frequent_list[j])
                if len(union) == len(prev_frequent_list[0]) + 1:
                    if union not in next_candidates:
                        next_candidates.append(union)
        return next_candidates

    def _generate_rules(self):
        """Generate association rules from frequent itemsets."""
        for k, itemsets in self.frequent_itemsets.items():
            if k < 2:
                continue
            for itemset in itemsets:
                for i in range(1, len(itemset)):
                    for antecedent in itertools.combinations(itemset, i):
                        antecedent = frozenset(antecedent)
                        consequent = itemset - antecedent
                        if consequent:
                            support = itemsets[itemset]
                            antecedent_support = self._get_support(antecedent)
                            confidence = support / antecedent_support if antecedent_support else 0
                            if confidence >= self.min_confidence:
                                self.rules.append({
                                    "antecedent": set(antecedent),
                                    "consequent": set(consequent),
                                    "support": round(support, 3),
                                    "confidence": round(confidence, 3)
                                })

    def _get_support(self, itemset):
        """Return support of an itemset if it exists."""
        for k, itemsets in self.frequent_itemsets.items():
            if itemset in itemsets:
                return itemsets[itemset]
        return 0

    def get_rules(self):
        """Return the generated association rules as a DataFrame."""
        return pd.DataFrame(self.rules)
