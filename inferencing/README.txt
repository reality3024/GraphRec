這個是我用toy dataset做出來的
原程式並沒有包含inference的結果
這邊我輸出了一個predictions.csv
資料說明如下:
user_id: 使用者	
item: 商品
predicted_rating_continuous: model輸出的原始結果	
predicted_rating_round: 我輸出的最接近的key值

eg. 程式會用rating list去查表找到最終的評分(也就是用key去找value)，這邊用toy dataset為例，總共有8種評分
0.5: 7
1.0: 1
1.5: 6
2.0: 0
2.5: 4
3.0: 2
3.5: 5
4.0: 3
假設我今天值是3.8，最靠近的值是4.0，那最終結果就是3

target_key: ground truth的rating list的key值	
predicted_rating: 我的model最終判定的評分	
target_label: 測試資料集的實際評分
