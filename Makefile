SHELL := bash
.ONESHELL:

data:
	mkdir $@

# Set up 
mappings:
	python scripts/dst/collect_jobs.py
	python scripts/dst/collect_kraf.py
	python scripts/dst/collect_mappings.py

clean_mappings:
	rm nsdata/times-mappings/*
	rm nsdata/hq-mappings/*

# The new one
nodes:
	python -m dstnx.data.school_cohort  \
		--start 1992 \
		--end 1996 \
		--suffix _new

# Klasseids 
grundskoleklassetable:
	python scripts/create_klasseid_table.py

# Depends on grundskoleklassetable
classids:
	python -m dstnx.data.school_classids --suffix _new

grades:
	python -m dstnx.data.grades --suffix _new

outcomes:
	python -m dstnx.outcomes.edu --suffix _new --max-age-edu 25
	python -m dstnx.outcomes.edu --suffix _new --max-age-edu 21

jobs:
	python -m dstnx.outcomes.job --suffix _new --max-age-job 24
	python -m dstnx.outcomes.job --suffix _new --max-age-job 20

kom:
	python -m dstnx.features.address --suffix _new

node_addresses:
	python scripts/create_address.py --suffix _new

# Table with geodata to be used in next make statement
geoaddresses:
	python scripts/create_geotable.py 

# Collect kraf for 1985-2020
kraf:
	python -m dstnx.data.kraf collect-all

own_kraf:
	python -m dstnx.data.kraf construct-kraf-cohort --suffix _new

psyk:
	python -m dstnx.data.psyk

# Construct the tables with features on whole pop for each year
neighbors_pop:
	python scripts/construct_neighbors.py  \
		--start 1985 \
		--end 2020 \
		--force \
		--geo-only


## Neighborhood measures for adults & parents
# RADIUS
upbringing_adults_radius:
	python -m dstnx.neighborhood.nearest --suffix _new \
		--neighbor-type neighbors \
		--file-suffix _adults \
		--spacing 1 \
		--batch-size 20000 \
		--parents

upbringing_youth_radius:
	python -m dstnx.neighborhood.nearest --suffix _new \
	--neighbor-type youth \
	--file-suffix _youth \
	--spacing 1 \
	--batch-size 20000

# K nearest
upbringing_adults_k:
	python -m dstnx.neighborhood.nearest --suffix _new \
		--neighbor-type neighbors \
		--file-suffix _adults \
		--spacing 1 \
		--batch-size 20000 \
		--k-nearest

upbringing_youth_k:
	python -m dstnx.neighborhood.nearest \
		--start 1999 \
		--end 2020 \
		--suffix _new \
		--neighbor-type youth \
		--file-suffix _youth \
		--spacing 1 \
		--batch-size 20000 \
		--k-nearest \
		--save-edges

upbringing_adults: upbringing_adults_radius upbringing_adults_k
upbringing_youth: upbringing_youth_radius upbringing_youth_k

# Aggregate measures 
agg_features_two:
	python -m dstnx.features.agg_features \
		--age-period two \
		--suffix _new_twoyear

agg_features_radius_default:
	python -m dstnx.features.agg_features neighbors \
		--age-period default \
		--suffix _new

agg_features_k_default:
	python -m dstnx.features.agg_features neighbors \
		--age-period default \
		--suffix _new \
		--k-nearest

agg_features_parents_default:
	python -m dstnx.features.agg_features parents \
		--age-period default \
		--suffix _new

agg_features_default: agg_features_radius_default agg_features_k_default agg_features_parents_default

# Depends on aggregated parent features
peer_ses:
	python -m dstnx.features.peers_ses --suffix _new

ninthgrade_insts:
	python -m dstnx.features.inst --suffix _new

# Merge measures
merge_full_radius:
	python -m dstnx.features.merge \
		--suffix _new \
		--radius 100 \
		--force \
		--save 

merge_full_k:
	python -m dstnx.features.merge \
		--suffix _new \
		--k 30 \
		--force \
		--save 

merge_full_large:
	python -m dstnx.features.merge \
		--suffix _new \
		--radius 600 \
		--force \
		--save 

estimate_linear:
	python -m dstnx.models.linear

estimate_k:
	python -m dstnx.models.linear --k 30 \
		--col-suffix _all


# Set paths for R 

paths:
	cmd /C scripts\windows\set_paths.bat

agg_features_k:
	python -m dstnx.features.agg_features $(AGG_TYPE) \
		--age-period $(AGE_PERIOD) \
		--suffix $(SUFFIX) \
		--k-nearest \
		--col-suffix $(COL_SUFFIX)

merge_full_k:
	python -m dstnx.features.merge \
		--suffix $(SUFFIX) \
		--k $(K) \
		--force \
		--save \
		--col-suffix $(COL_SUFFIX) \
		--feature-suffix $(FEATURE_SUFFIX)

r_script_k:
	python -m dstnx.models.rutils \
		--suffix $(SUFFIX) \
		--k $(K) \
		--col-suffix $(COL_SUFFIX) \
		--feature-suffix $(FEATURE_SUFFIX)
	echo "Running Rscript and piping output to txt file"
	Rscript scripts/R/fe_k$(K)$(COL_SUFFIX)$(FEATURE_SUFFIX)-quintiles.R > ../tables/r-terminal-output/fe_k$(K)$(COL_SUFFIX)$(FEATURE_SUFFIX)-quintiles.txt

r_script_k_deciles:
	python -m dstnx.models.rutils \
		--suffix $(SUFFIX) \
		--k $(K) \
		--col-suffix $(COL_SUFFIX) \
		--feature-suffix $(FEATURE_SUFFIX) \
		--deciles
	echo "Running Rscript and piping output to txt file"
	Rscript scripts/R/fe_k$(K)$(COL_SUFFIX)$(FEATURE_SUFFIX)-deciles.R > ../tables/r-terminal-output/fe_k$(K)$(COL_SUFFIX)$(FEATURE_SUFFIX)-deciles.txt

agg_features_radius:
	python -m dstnx.features.agg_features $(AGG_TYPE) \
		--age-period $(AGE_PERIOD) \
		--suffix $(SUFFIX) \
		--col-suffix $(COL_SUFFIX)

merge_full_radius:
	python -m dstnx.features.merge \
		--suffix $(SUFFIX) \
		--radius $(RADIUS) \
		--force \
		--save \
		--col-suffix $(COL_SUFFIX) \
		--feature-suffix $(FEATURE_SUFFIX)


r_script_radius:
	python -m dstnx.models.rutils \
		--suffix $(SUFFIX) \
		--radius $(RADIUS) \
		--col-suffix $(COL_SUFFIX) \
		--feature-suffix $(FEATURE_SUFFIX)
	echo "Running Rscript and piping output to txt file"
	Rscript scripts/R/fe_radius$(RADIUS)$(COL_SUFFIX)$(FEATURE_SUFFIX)-quintiles.R > ../tables/r-terminal-output/fe_radius$(RADIUS)$(COL_SUFFIX)$(FEATURE_SUFFIX)-quintiles.txt

r_script_radius_deciles:
	python -m dstnx.models.rutils \
		--suffix $(SUFFIX) \
		--radius $(RADIUS) \
		--col-suffix $(COL_SUFFIX) \
		--feature-suffix $(FEATURE_SUFFIX) \
		--deciles
	echo "Running Rscript and piping output to txt file"
	Rscript scripts/R/fe_radius$(RADIUS)$(COL_SUFFIX)$(FEATURE_SUFFIX)-deciles.R > ../tables/r-terminal-output/fe_radius$(RADIUS)$(COL_SUFFIX)$(FEATURE_SUFFIX)-deciles.txt

# Tables

tables:
	python -m dstnx.tables.interact
	python -m dstnx.tables.ml_res
	python -m dstnx.tables.descriptive
	python -m dstnx.tables.age_periods
	python -m dstnx.tables.class_sizes

# PLOTS

# This should be run first if we haven't constructed the neigh measures;
pca_all_year:
	python scripts/pca_all_years.py

pca_plot:
	python -m dstnx.plots.pca 

ses_bins_plot:
	python -m dstnx.plots.ses_bins 

pca_plots: pca_plot ses_bins_plot

rank_plot:
	python -m dstnx.plots.rankplots 

plots: pca_plots rank_plot

move:
	python -m dstnx.utils.move


make desc: tables plots move



# Utils 

# Follow tail of latest log
log:
	python scripts/log_tail.py

format:
	black src

# Wheels manually downloaded from https://data.pyg.org/whl/torch-2.0.0%2Bcu117.html
pyg:
	pip install pyg-lib torch-scatter --no-index -f nsdata/packages

tensorboard:
	tensorboard --logdir nsdata/lightning/lightning_logs --bind_all


# Other
time_network:
	python scripts/generate_testdata.py --size 1000 --suffix "" --large

time_network_large:
	python scripts/generate_testdata.py --size 5000 --suffix "_large" --large
