import polars as pl
import polars.selectors as cs
import hvplot
import hvplot.polars
import holoviews as hv
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

#matplotlib.use("pgf")
plt.rcParams.update({
    #"svg.fonttype": "none",
    "svg.fonttype": "none",
    #"pgf.texsystem": "pdflatex",  # or "lualatex", "xelatex" depending on your document
    #"font.family": "serif",
    #"text.usetex": True,
    #"font.size": 8,
    #"pgf.rcfonts": False,
})

hvplot.extension('matplotlib')

"""
sudo apt update
sudo apt install texlive-full
"""
# latex font update
"""
sudo updmap-sys --setoption pdftexDownloadBase14 true
sudo updmap-sys
"""

# sudo apt update
# sudo apt-get install texlive

#path =  './experiments/fed-v-fisher/*.ndjson'
schema = pl.read_ndjson('experiments/fed-v-fisher-final/*.ndjson').schema
path =  'experiments/fed-v-fisher*/*.ndjson'
#path =  './experiments/fed-v-fisher-final/*.ndjson'
df = pl.read_ndjson(path,schema=schema,ignore_errors=True)

#df.write_ndjson('./experiments/fed-v-fisher-final/results1.ndjson')


print(df.columns)
#df = df.with_columns(max_split_percentage=pl.col('split_percentages').list.max())

df = df.select('name', 'num_clients', 'num_samples', cs.ends_with('_p_values'))

df = df.with_columns(
    experiment_type=pl.col('name').str.slice(0,3),
    conditioning_type=pl.col('name').str.slice(4)
)

experiment_types = ['M-O', 'C-B', 'B-O', 'C-O', 'C-M']
type_idx = -1
if type_idx >= 0:
    current_experiment_type = experiment_types[type_idx]
    df = df.filter(pl.col('experiment_type') == current_experiment_type)
else:
    current_experiment_type = ''

df = df.filter(pl.col('num_samples') < 4000)
#df = df.filter(pl.col('num_client') <= 5)
df = df.filter(~(pl.col('name').str.contains('\(')))

print(df.group_by('experiment_type', 'conditioning_type').agg(pl.len()).sort('len'))
#df = df.sort('experiment_type', 'conditioning_type')
df = df.explode('federated_p_values', 'fisher_p_values', 'baseline_p_values')

num_samples = 2000
alpha = 0.05
plot = None
_df = df.filter(pl.col('num_samples') == num_samples)
_df = _df.rename({
    'federated_p_values': 'Federated',
    'fisher_p_values': 'Meta-Analysis'
})
for i in [1,3,5,7]:


    __df = _df.filter(pl.col('num_clients') == i)
    #print(i, len(_df), len(__df))
    __df = __df.sample(min(1_000, len(__df)))

    x = __df['Federated'].to_numpy()
    y = __df['Meta-Analysis'].to_numpy()

    slope, intercept = np.polyfit(x, y, 1)  # linear fit

    x_min, x_max = x.min(), x.max()
    y_min = intercept + slope * x_min
    y_max = intercept + slope * x_max

    best_fit = hv.Curve(
        [(x_min, y_min), (x_max, y_max)], 
        kdims=['Federated'], vdims=['Meta-Analysis']
    ).opts(color='red', line_width=2)

    _plot = __df.hvplot.scatter(
        x='Federated',
        y='Meta-Analysis',
        alpha=0.6,
        ylim=(-0.01,1.01),
        xlim=(-0.01,1.01),
        width=400,
        height=400,
        #by='Method',
        legend='bottom_right',
        #backend='matplotlib',
        s=5000,
        xlabel=r'Fed p-value',  # LaTeX-escaped #
        ylabel=r'Meta p-value',
        color='baseline_p_values',
        #linestyle=['dashed', 'dotted']
        #title=f'{"Client" if i == 1 else "Clients"}'
    )

    _plot = _plot * best_fit

    _render =  hv.render(_plot, backend='matplotlib')
    _render.savefig(f'images/pval_comp/scatter-2-c{i}-samples{num_samples}.svg', format='svg', bbox_inches='tight', dpi=300)


    __df = __df.filter((pl.col('Federated') < 0.101) & (pl.col('Meta-Analysis') < 0.101))

    x = __df['Federated'].to_numpy()
    y = __df['Meta-Analysis'].to_numpy()

    slope, intercept = np.polyfit(x, y, 1)  # linear fit

    x_min, x_max = x.min(), x.max()
    y_min = intercept + slope * x_min
    y_max = intercept + slope * x_max

    best_fit = hv.Curve(
        [(x_min, y_min), (x_max, y_max)], 
        kdims=['Federated'], vdims=['Meta-Analysis']
    ).opts(color='red', line_width=2)

    _plot = __df.hvplot.scatter(
        x='Federated',
        y='Meta-Analysis',
        alpha=0.6,
        ylim=(-0.01,0.1),
        xlim=(-0.01,0.1),
        width=400,
        height=400,
        #by='Method',
        legend='bottom_right',
        #backend='matplotlib',
        s=5000,
        xlabel=r'Fed p-value',  # LaTeX-escaped #
        ylabel=r'Meta p-value',
        color='baseline_p_values',
        #linestyle=['dashed', 'dotted']
        #title=f'{"Client" if i == 1 else "Clients"}'
    )

    _plot = _plot * best_fit

    _render =  hv.render(_plot, backend='matplotlib')
    _render.savefig(f'images/pval_comp/scatter-2-c{i}-samples{num_samples}-small.svg', format='svg', bbox_inches='tight', dpi=300)


