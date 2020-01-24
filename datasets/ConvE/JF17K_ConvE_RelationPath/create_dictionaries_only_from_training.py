import json, argparse, operator, os, re

entity2id_file = open('entity2id.txt', 'w')
relation2id_file = open('relation2id.txt', 'w')
entity2id = {}
relation2id = {}
entity_counter = 0
relation_counter = 0
with open("train.txt") as file:
    for line in file:
        splitted_line = line.strip().split()
        h = splitted_line[0]
        r = splitted_line[1]
        t = splitted_line[2]

        if h not in entity2id:
            entity2id[h] = entity_counter
            entity_counter += 1
        if t not in entity2id:
            entity2id[t] = entity_counter
            entity_counter += 1
        if r not in relation2id:
            relation2id[r] = relation_counter
            relation_counter += 1

for e in entity2id:
    entity2id_file.write(e + "\t" + str(entity2id[e]) + "\n")
for r in relation2id:
    relation2id_file.write(r + "\t" + str(relation2id[r]) + "\n")

entity2id_file.close()
relation2id_file.close()
file.close()


print("END OF SCRIPT")
