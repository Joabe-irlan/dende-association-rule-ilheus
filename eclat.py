import pandas as pd
from itertools import combinations



df = pd.read_csv("vendas_dataset.csv")


df = df.dropna(subset=['descricao_produtos'])


def limpar_e_separar(texto):
    marcas = ['nike', 'adidas', 'puma', 'zara']
    
    itens = texto.split(';')
    
    itens_limpos = []
    for item in itens:
        item = item.lower().strip()
        for marca in marcas:
            item = item.replace(marca, '')
        item = item.strip()
        if item: 
            itens_limpos.append(item)
    return itens_limpos

transactions = df['descricao_produtos'].apply(limpar_e_separar).tolist()

min_support = 0.005 
total_transactions = len(transactions)


vertical_data = {}

for tid, transaction in enumerate(transactions):
    for item in set(transaction):
        if item not in vertical_data:
            vertical_data[item] = set()
        vertical_data[item].add(tid)

frequent_itemsets = []

def eclat(prefix, items):
    while items:
        item, tids = items.pop()
        new_prefix = prefix + [item]
        support = len(tids) / total_transactions

        if support >= min_support:
            frequent_itemsets.append((new_prefix, support))
            new_items = []
            for other_item, other_tids in items:
                intersection = tids.intersection(other_tids)
                new_support = len(intersection) / total_transactions
                if new_support >= min_support:
                    new_items.append((other_item, intersection))
            
            eclat(new_prefix, new_items)

items_list = list(vertical_data.items())
eclat([], items_list)


print(f"Total de Transações Analisadas: {total_transactions}")
print("\n--- ITEMSETS FREQUENTES (Suporte >= {0:.0%}) ---".format(min_support))

for itemset, support in sorted(frequent_itemsets, key=lambda x: x[1], reverse=True):
    print(f"Itemset: {itemset} | Suporte: {support:.4f}")


print("\n--- REGRAS DE ASSOCIAÇÃO ---")
support_dict = {tuple(sorted(itemset)): support for itemset, support in frequent_itemsets}
min_confidence = 0.3

for itemset, support in frequent_itemsets:
    if len(itemset) < 2:
        continue
    
    itemset_sorted = sorted(itemset)
    for i in range(1, len(itemset_sorted)):
        for antecedent in combinations(itemset_sorted, i):
            antecedent = tuple(sorted(antecedent))
            consequent = tuple(sorted(set(itemset_sorted) - set(antecedent)))
            
            antecedent_support = support_dict.get(antecedent)
            if antecedent_support:
                confidence = support / antecedent_support
                if confidence >= min_confidence:
                    print(f"Regra: {antecedent} -> {consequent}")
                    print(f"Confiança: {confidence:.2f} | Suporte: {support:.4f}")
                    print("-" * 30)