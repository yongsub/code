import sys, os, json, time, wget

# CMD = 'wget "https://api.korbit.co.kr/v1/transactions?currency_pair={}&time=minute" -q -O data/tmp.json'
URL = 'https://api.korbit.co.kr/v1/transactions?currency_pair={}&time=minute'
TIME_INTERVAL = 1

# argv = sys.argv
argv = [None, 'eth_krw']

currency = argv[1]  # eth_krw or btc_krw
log_file = 'data/ethereum/' + currency + '_log'
log_fields = [u'timestamp', u'tid', u'price', u'amount']

err_log_file = log_file + '_err'

n_last_few = 1000
last_few_tids = []

while True:
    try:
        # this_cmd = CMD.format(currency)
        # os.system(this_cmd)
        wget.download(url=URL, out='tmp.json')

        with open('data/tmp.json', 'r') as f_tmp:
            transactions_json = f_tmp.read()

        transactions = json.loads(transactions_json)

        with open(log_file, 'a') as f:            
            new_transactions = filter(lambda tr: tr['tid'] not in last_few_tids, transactions)
            new_records = map(lambda new_tr: ','.join([str(new_tr[key]) for key in log_fields]), new_transactions)
        
            if len(new_records) > 0:
                new_records_in_string = '\n'.join(new_records[::-1])
                f.write(new_records_in_string + '\n')
                
                new_prices = [new_tr['price'] for new_tr in new_transactions]
                min_max = [int(min(new_prices)), int(max(new_prices))]
                min_max_diff = min_max + [min_max[1] - min_max[0]]
            else:
                min_max_diff = [None] * 3
    
            last_few_tids += [new_tr['tid'] for new_tr in new_transactions]
            last_few_tids = last_few_tids[-n_last_few:]
    except Exception as e:
        print e
    
    time.sleep(TIME_INTERVAL)