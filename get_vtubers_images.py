import os
import cv2
import requests
import argparse
import pandas as pd
from requests import exceptions

MAX_SIZE = 64
GROUP_SIZE = 10
MAX_RESULTS = 255

# 取得した画像の保存先
SAVE_DIR_PATH = './img/original_images/'

# 取得したエンドポイントURL
URL='https://api.bing.microsoft.com/v7.0/images/search'
EXCEPTIONS = set([IOError, FileNotFoundError,
                exceptions.RequestException, exceptions.HTTPError,
                exceptions.ConnectionError, exceptions.Timeout]
            )
# ここにBing Search API Keyを入力
HEADERS = {'Ocp-Apim-Subscription-Key' : 'xxxxx'}

# Vごとのディレクトリを作成する。
def get_image_dir(name):
    dirname = SAVE_DIR_PATH + name

    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    return dirname

# 検索結果を返却
def get_search_result(parameter):
    search = requests.get(URL, headers=HEADERS, params=parameter)
    search.raise_for_status()

    results = search.json()
    estNumResults = min(results['totalEstimatedMatches'], MAX_RESULTS)
    return search, results, estNumResults

def get_image(name,parameter):
    total = 0 # 画像を通し番号名にする
    search, results, estNumResults = get_search_result(parameter)

    for offset in range(0, estNumResults, GROUP_SIZE):
        print('[INFO] making request for group {}-{} of {}...'.format(offset, offset + GROUP_SIZE, estNumResults))
        parameter['offset'] = offset
        search = requests.get(URL, headers=HEADERS, params=parameter)
        search.raise_for_status()
        results = search.json()

        for v in results['value']:
            try:
                print('fetching: {}'.format(v['contentUrl']))

                # 拡張子を取得
                ext = os.path.splitext(v['contentUrl'])[1]
                
                # 画像の保存先を決定
                file_path = '{}/{}{}'.format(get_image_dir(name), str(total).zfill(8), ext)

                # 画像を書き込み
                total += 1
                with open(file_path, "wb") as img:
                    img.write(requests.get(v['contentUrl']).content)

            except Exception as e:
                if type(e) in EXCEPTIONS:
                    print('skip: {}'.format(v['contentUrl']))
                    continue
                else:
                    print('skip:System error')
                    continue

# 検索名リストのパス
target_csv_path = './target.csv'
target = pd.read_csv(target_csv_path)
for line in target.values:
    
    if line[1] == 1.0:
        continue
    
    print(line[0])
    parameter = {
        'q': line[0],
        'offset': 0,
        'count': GROUP_SIZE,
        'imageType':'Photo',
        'color':'ColorOnly'
    }

    get_image(line[0],parameter)

    target.loc[target['VtuberName'] == line[0], 'IgnoreFlag'] = 1
    target.to_csv(target_csv_path, index=False)
    