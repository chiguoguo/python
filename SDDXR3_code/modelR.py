#coding=utf-8
#feature,reversed order

demoline="13356614239,OKW-i160|140626,SCH-W609|140320,LT-L202|130426,SCH-M609|120322,SCH-W609|120207"
demoline2="15376012176,HW-C2829|160210,HW-C2829|150310,HW-C2829|150103,SCH-E189|141217,NOK-1325|130308"
demoline3="15376012176,HW-C2829|160210,HW-C2829|150310,H|150103,SCH-E189|141217,NOK-1325|130308"
demoline4="15376012176,HW-C2829|160210,HW-C2829|150310,H|150103"

from datetime import datetime
from pickle import load
import pandas as pd

class modelR:
    def __init__(self):
        self.source_file="change2.csv"
        self.base_day=datetime(2016,5,1)
        self.durations=[-600,60]
        self.step=30
        self.init_windows()
        self.init_phone_dict()


    def day2time(self,day):
        daystr_tuple=2000+int(day[:-4]),int(day[-4:-2]),int(day[-2:])
        return (datetime(*daystr_tuple)-self.base_day).days

    def isin_window(self,time,window):
        return 1 if window[0] <= time < window[1] else 0

    def init_windows(self):
        self.history_window,self.buy_window=(self.durations[0],0),(0,self.durations)
        self.history_windows=tuple((-i-self.step,-i) for i in range(0,-self.durations[0],self.step))
        self.n_history_windows=len(self.history_windows)
        #((-30, 0), (-60, -30), (-90, -60), (-120, -90), (-150, -120), (-180, -150), (-210, -180), (-240, -210), (-270, -240), (-300, -270), (-330, -300), (-360, -330), (-390, -360), (-420, -390), (-450, -420), (-480, -450), (-510, -480), (-540, -510), (-570, -540), (-600, -570)

    def init_phone_dict(self):
        with open("phone_dict.pickle","rb") as f:
            self.phone_dict=load(f)
            print("phone_dict is on")

    def search_phone_code(self,phone_type):
        brand,price=self.phone_dict[phone_type] if phone_type in self.phone_dict else "Zc"
        return ord(brand)-65,ord(price)-97   #brand_code, price

    def dummy_brands(self,brands):
        phone_onehot=[]
        for p in brands:
            tmp=[0]*26;tmp[p]=1
            phone_onehot.extend(tmp)
        # print len(phone_onehot),phone_onehot
        return phone_onehot

    def dummy_features(self,oneline):
        oneline=oneline.strip().split(",")
        user,oneline=oneline[0],oneline[1:]
        phones=[]
        for item in oneline:
            item=item.split("|")
            time=self.day2time(item[1])
            if time>=0:continue
            time_index=-time//self.step+1
            if time_index >= self.n_history_windows:
                phones.extend([self.search_phone_code(item[0])]*(self.n_history_windows-len(phones)))
                break
            else:
                phones.extend([self.search_phone_code(item[0])]*(time_index-len(phones)))
        if len(phones) < self.n_history_windows:
            phones.extend([self.search_phone_code("Zc")]*(self.n_history_windows-len(phones)))
        #print phones
        brands,prices=[p[0] for p in phones],[p[1] for p in phones]
        # print brands,prices
        brands=self.dummy_brands(brands)
        # print len(brands),brands

        times=[self.day2time(item.split("|")[1]) for item in oneline]
        # print times
        month_onehot=[0]*len(self.history_windows)
        for time in times:
            for i,window in enumerate(self.history_windows):
                if self.isin_window(time,window):month_onehot[i]+=1
        # print month_onehot
        month_onehot_real=[1 if one else 0 for  one in month_onehot]
        month_accumlate=[0]*len(self.history_windows)
        accu=0
        for i,v in enumerate(month_onehot):
            accu+=v
            month_accumlate[i]=accu
        # print month_accumlate
        month_accumlate_real=[0]*len(self.history_windows)
        accu=0
        for i,v in enumerate(month_onehot_real):
            accu+=v
            month_accumlate_real[i]=accu
        is_buy=int(bool(sum([self.isin_window(time,self.buy_window) for time in times])))
        # print is_buy

        return user,is_buy,month_onehot,month_onehot_real,month_accumlate,month_accumlate_real,prices,brands

    def init_cols(self,first_line):
        self.feature_dict={"user":[],"is_buy":[],}
        user,is_buy,month_onehot,month_onehot_real,month_accumlate,month_accumlate_real,prices,brands=\
                self.dummy_features(first_line)
        for i in range(len(month_onehot)):
            self.feature_dict["m_%s"%i]=[]
            self.feature_dict["mr_%s"%i]=[]
            self.feature_dict["ma_%s"%i]=[]
            self.feature_dict["mar_%s"%i]=[]
            self.feature_dict["p_%s"%i]=[]
        for i in range(len(brands)):
            self.feature_dict["b_%s"%i]=[]
        # print(self.feature_dict)

    def mk_feature(self):
        with open(self.source_file,"rt") as f:
            lines=f.readlines()
            print("data source is on")
        self.init_cols(lines[0])
        count=0
        for line in lines[3000000:5000000]:
            count+=1
            if not count%100000:print count
            user,is_buy,month_onehot,month_onehot_real,month_accumlate,month_accumlate_real,prices,brands=\
                self.dummy_features(line)
            self.feature_dict["user"].append(user)
            self.feature_dict["is_buy"].append(is_buy)
            for i,v in enumerate(zip(month_onehot,month_onehot_real,month_accumlate,month_accumlate_real,prices)):
                self.feature_dict["m_%s"%i].append(v[0])
                self.feature_dict["mr_%s"%i].append(v[1])
                self.feature_dict["ma_%s"%i].append(v[2])
                self.feature_dict["mar_%s"%i].append(v[3])
                self.feature_dict["p_%s"%i].append(v[4])
            for i,v in enumerate(brands):
                self.feature_dict["b_%s"%i].append(v)
        print("feature_dict is on")
        self.df_feature=pd.DataFrame(self.feature_dict)
        order_columns=["user","is_buy"]+sorted(list(set(self.df_feature.columns)-{"user","is_buy"}))
        self.df_feature=self.df_feature.reindex(columns=order_columns)
        print("dumping feature dataframe...")
        self.df_feature.to_csv("feature_sd_05_35.csv",encoding="utf-8",index=False)
        print("dumped.")

if __name__ == '__main__':
    m=modelR()
    # print(m.history_windows)
    # print(m.dummy_window(demoline2))
    # print(m.phone_dict)
    m.mk_feature()





