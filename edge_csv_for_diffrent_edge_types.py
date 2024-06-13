#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:19:04 2022

@author: saikat
"""

import pandas as pd

edges_df = pd.read_csv("/home/saikat/OpenHGNN_test/My_Hetero_net_data_T2DM_dummy/final_fully_weighted_hetero_net.csv")

edges_df = edges_df.drop(columns=['bidirect'])

# edges_df.edge_type_id = edges_df.edge_type_id.astype(int)

new_id_map = pd.read_csv('/home/saikat/OpenHGNN_test/My_Hetero_net_data_T2DM_dummy/new_to_old_id_map_table.csv')

old_to_new_id_dict = dict(zip(new_id_map.ID, new_id_map.new_ID))

edges_df['entity1'] = edges_df['entity1'].map(old_to_new_id_dict)
edges_df['entity2'] = edges_df['entity2'].map(old_to_new_id_dict)



edges_df['edge_type_id'] = edges_df['edge_type_id'].astype('category').cat.codes

edges_df = edges_df.sort_values(by = 'edge_type_id', ascending=True)

## making df for edge type to edge type id mappings
edge_type_id_to_edge_type = edges_df.drop(columns=['entity1','entity2','edge_weight'])

edge_type_id_to_edge_type = edge_type_id_to_edge_type[['edge_type_id','edge_type']]

edge_type_id_to_edge_type = edge_type_id_to_edge_type.drop_duplicates()

edge_type_id_to_edge_type.to_csv("/home/saikat/OpenHGNN_test/My_Hetero_net_data_T2DM_dummy/edge_type_to_new_id_map.csv", index=False)

########################################################################

edges_df = edges_df.drop(columns=['edge_type'])

edges_df[['entity1','entity2','edge_type_id']] = edges_df[['entity1','entity2','edge_type_id']].astype(int)

edges_df['edge_weight'] = edges_df['edge_weight'].astype(float)


for (edge_type), group in edges_df.groupby(['edge_type_id']):
    group.to_csv(f'T2DM_hetero_edges/edges_{edge_type}.csv', index=False)