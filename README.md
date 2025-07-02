# Invest Analyzer

## システム概要

Invest Analyzerは、株価データの管理と予測値の保存を目的としたDjangoアプリケーションです。

Dockerを利用してDjangoのWebサーバとMySQLデータベースをコンテナで構築しました。

## 注意事項

本アプリは機械学習の学習を目的として開発されたものであり、株価予測の結果について一切の保証はいたしかねます。

株の売買や投資の判断は、ご自身の責任でお願いいたします。

### 主な機能

- 株価情報（証券コード、企業名、日付、終値、予測値）を管理  
- [yfinance](https://pypi.org/project/yfinance/)ライブラリを使ってYahoo Financeから株価データを自動取得
- 機械学習モデルにより株価の予測値を生成し保存
- Django管理画面からデータの閲覧・編集が可能
- MySQLデータベースにデータを保存し、高速な検索を実現

---

## 開発環境構成

- Python 3.10
- Django 4.2
- MySQL 8 (Dockerコンテナ)
- Docker & Docker Compose
- yfinance (株価データ取得用)
- matplotlib
- pandas
- numpy
- PyTorch（RNN/LSTMモデル用）  
- japan_stocks.csv（日本株の企業名・コードデータ）

---

## 起動方法

プロジェクトディレクトリへ移動
```bash
cd invest_analyzer
```

Dockerコンテナをビルド・起動
```bash
docker-compose up -d --build
```

Djangoのマイグレーションを実行（モデル変更をデータベースに反映）
```bash
docker-compose exec web python manage.py makemigrations
docker-compose exec web python manage.py migrate
```

管理画面にアクセスするための管理ユーザーを作成（任意）
```bash
docker-compose exec web python manage.py createsuperuser
```

ブラウザでDjangoアプリにアクセス

アプリ本体

http://localhost:8000/


管理画面

http://localhost:8000/admin/


MySQLデータベースに接続する方法

MySQLのコンテナ内に入ってSQL操作したい場合：
```bash
docker-compose exec db mysql -u user -p
```
パスワードは.envのDB_PASSWORDに設定したものを入力してください。



コンテナ停止・再起動

開発環境を停止したい場合：
```bash
docker-compose down -v
```

webコンテナを再度起動する場合：
```bash
docker-compose restart web
```

再ビルドする場合：
```bash
docker-compose up -d --build
```


DjangoのWebサーバーのログを確認
```bash
docker-compose logs web
```

MySQLデータベースのログを確認
```bash
docker-compose logs db
```