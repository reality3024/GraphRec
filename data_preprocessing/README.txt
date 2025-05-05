這裡是Epinion的資料集轉換方法
我把資料都分好了，可以跑create_pickle.py把.mat檔轉換成.pickle，如果要改變train/test比例需要從這邊修改
另外我把資料可視化成excel檔可以查看確認，資料集是從https://www.cse.msu.edu/~tangjili/trust.html來的
Epinion分類.txt是根據所有產品別做大項分類，不過在原文並沒有使用到這個feature。
你可以用read_pickle.ipynb來讀取pickle檔看裡面長啥樣子

rating.mat

rating.mat includes the rating information. there are five columns and they are userid, productid, categoryid, rating, helpfulness, respectively. 

*****************************************************************
For example, for one row
(1,2,3,4,5)

It means that user 1 gives a rating of 4 to the product 2 from the category 3. The helpfulness of this rating is 5. 

*****************************************************************************


===============================================================================================
trustnetwork.mat

trustnetwork.mat includes the trust relations between users. There are two columns and both of them are userid.

*************************************************************************************
for example, for one row,
(1,2)

it means that user 1 trusts user 2.
*************************************************************************************



----------------------------------------------------------------------------------------------------------------
References
----------------------------------------------------------------------------------------------------------------


@Conference{tang-etal12a,
  title={m{T}rust: {D}iscerning multi-faceted trust in a connected world},
  author={Tang, J. and Gao, H. and Liu, H.},
  booktitle={Proceedings of the fifth ACM international conference on Web search and data mining},
  pages={93--102},
  year={2012},
  organization={ACM}
}


@Confernce{tang-etal12b,
  title={e{T}rust: {U}nderstanding trust evolution in an online world},
  author={Tang, J. and Gao, H. and Liu, H. and Das Sarma, A.},
  booktitle={Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining},
  pages={253--261},
  year={2012},
  organization={ACM}
}