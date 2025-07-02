from django.db import models

# Create your models here.

class StockPrice(models.Model):
    code = models.CharField(max_length=10)         # 証券コード（例: 7203.T）
    name = models.CharField(max_length=64)         # 企業名
    date = models.DateField()                      # 日付
    close = models.FloatField()                    # 終値
    prediction = models.FloatField(null=True, blank=True)  # 予想値（任意）

    class Meta:
        unique_together = ('code', 'date')         # 同じ銘柄・日付の重複保存を防ぐ

    def __str__(self):
        return f"{self.name}({self.code}) {self.date}: {self.close} / 予想値: {self.prediction}"
