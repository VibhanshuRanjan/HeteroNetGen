#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 21:36:16 2022

@author: saikat

"""

import pandas as pd

df1_node_info=pd.read_csv("/home/saikat/OpenHGNN_test/My_Hetero_net_data_T2DM_dummy/final_entity_id_map_T2DM.csv")

df2_edge_info=pd.read_csv("/home/saikat/OpenHGNN_test/My_Hetero_net_data_T2DM_dummy/final_fully_weighted_hetero_net.csv")

df1_node_info = df1_node_info.drop(columns=['entity'])

df1_node_info=df1_node_info.rename(columns = {'ID':'entity1'})


df1_merge_df2_1 = pd.merge(df2_edge_info, df1_node_info, how="inner", on=['entity1']).rename(columns={'entity_type':'entity1_type'})


df1_node_info=df1_node_info.rename(columns = {'entity1':'entity2'})


df1_merge_df2_2 = pd.merge(df1_merge_df2_1, df1_node_info, how="inner",
                               on=['entity2']).rename(columns={'entity_type':'entity2_type'})

final_hetero_df = df1_merge_df2_2[['entity1','entity1_type','entity2','entity2_type','Edge_weight','Edge_type']]

final_hetero_df.to_csv("/home/saikat/OpenHGNN_test/My_Hetero_net_data_T2DM_dummy/final_updated_weighted_heteronet_with_types.csv")