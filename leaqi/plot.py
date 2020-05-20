import warnings
from scipy.interpolate import interp1d
import glob
import pandas as pd
import numpy as np
from string import Template
import seaborn as sns
from itertools import cycle
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as tick
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter, NullFormatter
import matplotlib.patheffects as mpe
import matplotlib.ticker as ticker

outline=mpe.withStroke(linewidth=15, foreground='black')

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['axes.linewidth'] = 2 #set the value globally

color_list = np.array(sns.color_palette("colorblind", n_colors=6))

parser = argparse.ArgumentParser()
parser.add_argument("--env", choices=['Ner-v0', 'Keyphrase-v0','Pos-v0'])
args = parser.parse_args()

file_name = Template('output/${env}/${env}_${alg}_${diff_clf_lr}_${diff_clf_th}_${model_lr}_${b}_${type}_${query_strategy}_${no_apple_tasting}_${betadistro}_${ref_type}*.csv')

leaqis = []
dagger_w_passives = []
dagger_s_passives = []
dagger_s_actives = []

# Leaqi param sweep
params = [({'env':args.env, 'alg':alg, 'diff_clf_lr':diff_clf_lr, 'diff_clf_th':diff_clf_th, 'model_lr':model_lr, 'b':b, 'type':type, 'query_strategy':query_strategy, 'no_apple_tasting':no_apple_tasting, 'betadistro':betadistro, 'ref_type':ref_type})
    #All params:
    #for model_lr in ['1e-6']
    #for diff_clf_th in ['.50']
    #for diff_clf_lr in ['1e-2', '1e-3','1e-4']
    #for b in [ '5e-1', '10e-1', '15e-1']
    #for type in ['normal']
    #for betadistro in ['1']
    #for no_apple_tasting in [True, False]
    #for alg in ['leaqi']
    #for query_strategy in ['active']]

    #Main plot params:
    for model_lr in ['1e-6']
    for diff_clf_th in ['.50']
    for diff_clf_lr in ['1e-2']
    for b in [ '5e-1']
    for type in ['normal']
    for betadistro in ['1']
    for no_apple_tasting in [False]
    for alg in ['leaqi']
    for ref_type in ['normal']
    for query_strategy in ['active']]

# Leaqi param sweep
params += [({'env':args.env, 'alg':alg, 'diff_clf_lr':diff_clf_lr, 'diff_clf_th':diff_clf_th, 'model_lr':model_lr, 'b':b, 'type':type, 'query_strategy':query_strategy, 'no_apple_tasting':no_apple_tasting, 'betadistro':betadistro, 'ref_type':ref_type})
    #All params:
    #for model_lr in ['1e-6']
    #for diff_clf_th in ['.50']
    #for diff_clf_lr in ['1e-2', '1e-3','1e-4']
    #for b in [ '5e-1', '10e-1', '15e-1']
    #for type in ['normal']
    #for betadistro in ['1']
    #for no_apple_tasting in [True, False]
    #for alg in ['leaqi']
    #for query_strategy in ['active']]

    #Main plot params:
    for model_lr in ['1e-6']
    for diff_clf_th in ['.50']
    for diff_clf_lr in ['1e-2']
    for b in [ '5e-1']
    for type in ['normal']
    for betadistro in ['1']
    for no_apple_tasting in [True, False]
    for alg in ['leaqi']
    for ref_type in ['normal','random']
    for query_strategy in ['active']]

## Expert param sweep
#params += [({'env':args.env, 'alg':alg, 'diff_clf_lr':None, 'diff_clf_th':None, 'model_lr':model_lr, 'b':b, 'type':type, 'query_strategy':query_strategy, 'no_apple_tasting':False, 'betadistro':None,'ref_type':ref_type})
#    #All params
#    #for model_lr in ['1e-6']
#    #for b in ['5e-1', '10e-1', '15e-1']
#    #for type in ['normal','weak_feature']
#    #for alg in ['dagger:strong']
#    #for query_strategy in ['active', 'random', 'passive']]
#
#    #Main plot params
#    for model_lr in ['1e-6']
#    for b in ['5e-1']
#    for type in ['normal','weak_feature']
#    for alg in ['dagger:strong']
#    for ref_type in ['normal']
#    for query_strategy in ['active']]
#
## Expert param sweep
#params += [({'env':args.env, 'alg':alg, 'diff_clf_lr':None, 'diff_clf_th':None, 'model_lr':model_lr, 'b':b, 'type':type, 'query_strategy':query_strategy, 'no_apple_tasting':False, 'betadistro':None,'ref_type':ref_type})
#    #All params
#    #for model_lr in ['1e-6']
#    #for b in ['5e-1', '10e-1', '15e-1']
#    #for type in ['normal','weak_feature']
#    #for alg in ['dagger:strong']
#    #for query_strategy in ['active', 'random', 'passive']]
#
#    #Main plot params
#    for model_lr in ['1e-6']
#    for b in ['5e-1']
#    for type in ['normal','weak_feature']
#    for alg in ['dagger:strong']
#    for ref_type in ['normal']
#    for query_strategy in ['passive']]

## Reference param sweep --------------------------------------------------------------
# Pos-v0:
# - accuracy: 0.06969711288569673
# - recall: 0.05152236437568184
# - precision: 0.3297365915582355
# - f1 score: 0.08911952652571084
# - avg lens: 25.46690734055355
# - num tags: 42326
# Keyphrase-v0:
# ** Old-Generated
# - f1: 0.27696177062374244
# - recall: 0.4480794270833333
# - precisio: 0.20042224810716366
# - f1 score: 0.27696177062374244
# - num_sents: 2809
# - avg lens: 26.262370950516196
# - num tags: 73771
# ** New Generated **
# - f1: 0.2566046404778314
# - recall: 0.42158897905265147
# - precisio: 0.18442995129200032
# - f1 score: 0.2566046404778314
# - num_sents: 2416
# - avg sent lenth: 151.19701986754967
# Ner-v0:
# - recall: 0.27518648718168565
# - precisio: 0.8835512732278046
# - f1 score: 0.41966591481154586
# - avg lens: 14.501887329962253
# - num tags: 203621

def reformat_large_tick_values(tick_val, pos):
    """
    Turns large tick values (in the billions, millions and thousands) such as 4500 into 4.5K and also appropriately turns 4000 into 4K (no zero after the decimal).
    """
    if tick_val >= 1000000000:
        val = round(tick_val/1000000000, 1)
        new_tick_format = '{:}B'.format(val)
    elif tick_val >= 1000000:
        val = round(tick_val/1000000, 1)
        new_tick_format = '{:}M'.format(val)
    elif tick_val >= 1000:
        val = round(tick_val/1000, 1)
        new_tick_format = '{:}K'.format(val)
    elif tick_val < 1000:
        new_tick_format = round(tick_val, 1)
    else:
        new_tick_format = tick_val

    # make new_tick_format into a string value
    new_tick_format = str(new_tick_format)

    # code below will keep 4.5M as is but change values such as 4.0M to 4M since that zero after the decimal isn't needed
    index_of_decimal = new_tick_format.find(".")

    if index_of_decimal != -1:
        value_after_decimal = new_tick_format[index_of_decimal+1]
        if value_after_decimal == "0":
            # remove the 0 after the decimal point since it's not needed
            new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal+2:]

    return new_tick_format

for param in params:
    folder = file_name.substitute(**param)
    results = []
    for idx, file in enumerate(glob.glob(folder)):
        df = pd.read_csv(file)
        df['idx'] = idx
        results.append(df)

    if not results:
        continue

    results = pd.concat(results)
    if param['alg'] == 'dagger:strong':
        if param['query_strategy'] == 'active':
            name = 'ActiveDAgger'
        elif param['query_strategy'] == 'random':
            name = 'ActiveRandDAgger'
        elif param['query_strategy'] == 'passive':
            name = 'DAgger'
        if param['type'] == 'weak_feature':
            name += '+Feat.'

        if param['query_strategy'] == 'passive':
            dagger_s_passives.append((results, param['model_lr'], name))
        else:
            dagger_s_actives.append((results, param['model_lr'], name))
    elif param['alg'] == 'leaqi':
        if param['ref_type'] == 'normal' and not param['no_apple_tasting']:
            name = f'LeaQI'
        elif  param['ref_type'] == 'random' and param['no_apple_tasting']:
            name = f'LeaQI+NoisyRef'
        elif param['ref_type'] == 'normal' and param['no_apple_tasting']:
            name = f'LeaQI+NoAT'
        else:
            continue
        leaqis.append((results, param['model_lr'], name))
    elif param['alg'] == 'dagger:weak':
        name ='PASSREF.'
        dagger_w_passives.append((results, param['model_lr'], name))
    else:
        raise Exception('Unknown alg type')

def interpolate(df, y_axis, x_axis='x', label=None):
    min_x = df[x_axis].min()
    max_x = df[x_axis].max()
    targetx = range(min_x, max_x, 1)
    resultsy = []
    resultsx = []
    for name, group in df.groupby(['idx']):
        group = group.drop_duplicates(subset=x_axis, keep="last")
        xdata = group[x_axis].tolist()
        ydata = group[y_axis].tolist()
        # interpolation function for the original data
        try:
            f = interp1d(xdata, ydata, bounds_error=False, fill_value=(ydata[0], ydata[-1]))
            targety = f(targetx)
            resultsy.append(targety)
            resultsx.append(targetx)
        except:
            pass
    return (np.array(resultsy), list(resultsx[0]), label)

def plot_metrics(filename=None, items=None, x_axis=None, y_axis=None,\
        title=None, line_value=None, ax=None):
   alpha = .9
   width = 10
   f, ax = plt.subplots(figsize=(20, 20))

   if y_axis in ['accuracy']:
       y_axis = 'difference classifier accuracy'


   if y_axis in ['model_f1', 'model_acur']:
       #ax.set_xscale('log')
       y_axis = 'phrase-label f-score' if y_axis  != 'model_acur' else 'accuracy'

   #if y_axis in ['number of points queried']:
   #     ax.set(ylim=(0, 100000))

   colors = cycle(['#7570b3','#1b9e77', '#d95f02'])
   linestyle = cycle(('o-', 'o-', '^-'))
   #linestyle = cycle(('o-', 'P-', '^-','o-', 'P-', 'o-', 'P-'))
   #colors = cycle(['#7570b3', '#1b9e77','#1b9e77', '#d95f02','#d95f02'])

   markevery = cycle([300,500,700])
   #lineStyle =  cycle(('-','--', '-','-', '--', '--'))
   #linestyle = cycle(('-'))

   n = 100
   max_y = 0
   max_x = 0
   for item in items:
       if isinstance(item, float):
             ax.hlines(item, 0, 10000, linestyles='dashed', linewidth=width, label='Ref.')
       elif item is not None:
           for (rcpo_y, rcpo_x, label) in item:
               color = next(colors)
               mu1 = rcpo_y.mean(axis=0)
               sigma1 = rcpo_y.std(axis=0)
               ax.plot(rcpo_x[::n], (mu1+sigma1)[::n], color=color, linestyle='--', linewidth=2,zorder=0)
               ax.plot(rcpo_x[::n], (mu1-sigma1)[::n], color=color, linestyle='--', linewidth=2,zorder=0)
               ax.fill_between(rcpo_x[::n], (mu1+sigma1)[::n], (mu1-sigma1)[::n], alpha=0.4, zorder=5, color=color)
               ax.plot(rcpo_x[::n], mu1[::n], next(linestyle), linewidth=width, label=label, color=color,markevery=next(markevery), markersize=30, markeredgewidth=10, path_effects=[outline], zorder=10)

               max_y = max(max_y, np.max((mu1+sigma1)[::n]))
               max_x = max(max_x, np.max(rcpo_x[::n]))

   if y_axis in ['number of words queried']:
       ax.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))

   if y_axis in ['phrase-label f-score']:
       if args.env in ['Ner-v0', 'Keyphrase-v0']:#
           ax.legend(fontsize=75)


   title_lookup = {'Pos-v0': 'Part of Speech Tagging',
                   'Ner-v0': 'Named Entity Recognition',
                   'Keyphrase-v0': 'Keyphrase Extraction'}

   ax.set_xlabel(x_axis, fontsize=65)
   ax.set_ylabel(y_axis, fontsize=65)
   ax.tick_params(axis='both', which='major', labelsize=65)
   ax.set_title(f'{title_lookup[args.env]}', fontsize=70)
   ax.grid(axis='both', which='minor', linestyle=':', linewidth='0.5', color='grey')
   ax.grid(axis='both', which='major', linestyle='-', linewidth='1.0', color='black')
   ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
   ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

   ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))

   with warnings.catch_warnings():
       warnings.simplefilter("ignore")
       plt.tight_layout()
       plt.savefig(f'{filename}', format='pdf',  bbox_inches = 'tight', pad_inches = 0)
       #ax.figure.savefig(f'{filename}')#, format='png',  bbox_inches = 'tight', pad_inches = 0)

   plt.close(f)

def plot_loop(type, x_axis=None, line_value=None, show_ref=False,\
       title=None, x_axis_title=None, y_axis_title=None, filename=None):
    leaqi_items = []
    for (item, model_lr, diff_clf_lr) in leaqis:
        label = f'{diff_clf_lr}'
        try:
            leaqi_items.append(interpolate(item, type,x_axis=x_axis, label=label))
        except:
            continue

    active_items = []
    if type != 'diff_clf_acur':
        for (item, model_lr, diff_clf_lr) in dagger_s_actives:
            label = f'{diff_clf_lr}'
            active_items.append(interpolate(item, type,x_axis=x_axis, label=label))

    passive_s_items = []
    if type != 'diff_clf_acur':
        for (item, model_lr, diff_clf_lr) in dagger_s_passives:
            label = f'{diff_clf_lr}'
            try:
                passive_s_items.append(interpolate(item, type,x_axis=x_axis, label=label))
            except:
                continue


    passive_w_items = None
    #if show_ref:
    #    passive_w_items = ref['Pos-v0']

    plot_metrics(
      filename=filename,\
      title=title,\
      x_axis=x_axis_title,\
      y_axis=y_axis_title,\
      items = [leaqi_items, passive_s_items, passive_w_items,active_items])

# seen vs queries --------------------------------------------
plt.clf()
plot_loop('expert_queries',\
          x_axis='x',\
          filename=f'{args.env}_queries.pdf',\
          title=f'{args.env} Queried vs Seen',\
          x_axis_title="number of words seen",\
          y_axis_title="number of words queried")


# queried vs metric type -------------------------------------
#for plot_metric in leaqis[0][0].columns:
for plot_metric in ['model_f1', 'model_acur']:
    import pdb; pdb.set_trace()
    #if plot_metric not in ['model_f1', 'model_acur']:
    #    continue
    #print(f' plot_metric: {plot_metric}')

    plt.clf()
    plot_loop(plot_metric,\
              x_axis='expert_queries',\
              filename=f'{args.env}_queried_test_{plot_metric}_error.pdf',\
              title=f'{args.env} {plot_metric} vs queried',\
              x_axis_title="number of words queried",\
              y_axis_title=f"{plot_metric}",
              show_ref=True)

# diff classifier accuracy -----------------------------------
plt.clf()
leaqi_diffs = []
for (item, model_lr, diff_clf_lr) in leaqis:
    label = f'Leaqi df_lr={diff_clf_lr}'
    leaqi_diffs.append(interpolate(item, 'diff_clf_f1', label=label))

plot_metrics(
  filename=f'{args.env}_diff_clf.pdf',\
  title=f'{args.env} Difference Classifier',\
  x_axis="number of words seen",\
  y_axis="difference classifier f-score",\
  items=[leaqi_diffs])
