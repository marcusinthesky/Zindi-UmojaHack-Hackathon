#%%
import holoviews as hv
import hvplot.pandas

hv.extension('bokeh')

#%%
X = context.io.load('xente_train')

# %%
Y = context.io.load('xente_sample_submission')

# %%
