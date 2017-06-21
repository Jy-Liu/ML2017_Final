import pdb
import sys
import traceback
import argparse
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_data(train_feature, train_label):
    features = pd.read_csv(train_feature,  header=0)
    labels = pd.read_csv(train_label, header=0)
    return pd.concat([features, labels], axis=1)


def main():
    parser = argparse.ArgumentParser(description='Plot spatial distribution ')
    parser.add_argument('train_feature', type=str, help='feature_train.csv')
    parser.add_argument('train_label', type=str, help='feature_train.csv')
    args = parser.parse_args()

    train = read_data(args.train_feature,
                      args.train_label)
    train = train[train['longitude'] > 1]

    tiler = cimgt.GoogleTiles(style='terrain')
    mercator = tiler.crs
    fig = plt.figure(dpi=300, figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1,
                         projection=mercator)

    # plot map
    ax.add_image(tiler, 6)
    ax.coastlines(resolution='50m', color='black', linewidth=1)
    ax.add_feature(cfeature.BORDERS)

    # sample from data
    samples = train.sample(frac=0.1).reset_index(drop=True)
    longitude = samples['longitude']
    latitude = samples['latitude']
    status = samples['status_group']

    # plot them on the map
    for i in range(len(samples)):
        colors = {'functional': 'go',
                  'non functional': 'ro',
                  'functional needs repair': 'bo'}
        ax.plot(longitude[i], latitude[i],
                colors[status[i]], markersize=0.5,
                transform=ccrs.Geodetic())
    ax.set_extent([longitude.min() - 0.5, longitude.max() + 0.5,
                   latitude.min() - 0.5, latitude.max() + 0.5])

    red_dot = plt.plot(z, "ro", markersize=15)
    # ax.legend()
    plt.savefig('/tmp/t.png')

if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
