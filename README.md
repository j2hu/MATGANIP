# MATGANIP
Generative Adversarial Networks Assisted Discovery of the Structure-Property Relationship in Off-stoichiometry Perovskites

********************
In the floders of results_MATGANIP_E and results_MATGANIP_F, these data points are used to evaluate the trained model.
In the floders of datafile_MATGANIP_E and datafile_MATGANIP_F, their structural files are given with the format of compressed file.
In the floder of SQLites, the file of MATGANIP_e is database of the DFT calculation and the information of POSCAR used in the MATGANIP.e, where 500 records are contained in the train table, and 6166 records are contained in the test table.
In the floder of double_perovskite_in_the_heatmap, there are a pynb file used to generate the heatmap and a comperssed file contained all of the DFT results of double perovskits.
******************

When we need load the machine learning model, we could load the MATGANIP.e with the load_MATGANIP_E.py file and MATGANIP.f with the load_MATGANIP_F.py file, respectively.
For the trained machine learning, these files with the format of pkl have given the trained parameters that are needed to load the trained MATGANIP.
