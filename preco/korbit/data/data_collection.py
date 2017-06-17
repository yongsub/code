import sys, os, json, time, urllib, wget, datetime

currency = sys.argv[1] # target \in [eth_krw, btc_krw]

TMP_JSON = 'tmp/{}.json'.format(currency)
CMD = 'wget "https://api.korbit.co.kr/v1/transactions?currency_pair={}&time=minute" --no-check-certificate -q -O {}'
TIME_INTERVAL = 20

log_file_prefix = '{}/{}'.format(currency, currency)
log_fields = [u'timestamp', u'tid', u'price', u'amount']

n_last_few = 10000
last_few_tids = []

while True:
    try:
        now = datetime.datetime.now().strftime('%y-%m-%d')

        this_cmd = CMD.format(currency, TMP_JSON)
        os.system(this_cmd)

        with open(TMP_JSON, 'r') as f_tmp:
            transactions_json = f_tmp.read()

        transactions = json.loads(transactions_json)

        log_file = '{}_{}'.format(log_file_prefix, now) 
        with open(log_file, 'a') as f:            
            new_transactions = filter(lambda tr: tr['tid'] not in last_few_tids, transactions)
            new_records = map(lambda new_tr: ','.join([str(new_tr[key]) for key in log_fields]), new_transactions)
        
            if len(new_records) > 0:
                new_records_in_string = '\n'.join(new_records[::-1])
                f.write(new_records_in_string + '\n')

                new_prices = map(lambda new_tr: float(new_tr['price']), new_transactions)    
                count_min_max_mean = len(new_prices), min(new_prices), max(new_prices), sum(new_prices)/len(new_prices)
            else:
                count_min_max_mean = 0, None, None, None
  
            last_few_tids += [new_tr['tid'] for new_tr in new_transactions]
            last_few_tids = last_few_tids[-n_last_few:]

        print '{}: \n\t{} records\n\tmin {}, \n\tmax {}, \n\tmean {}\n'.format(datetime.datetime.now(), *count_min_max_mean)
    except Exception as e:
        print e
        
    time.sleep(TIME_INTERVAL)