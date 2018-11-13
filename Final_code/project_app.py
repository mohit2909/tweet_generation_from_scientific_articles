import tweepy
import time
import os
import tensorflow as tf
import pickle
from model import Model
from utils import build_dict, build_dataset, batch_iter
def setup():
    """setting up the OAuth connection."""

    # setting up the OAuth
    consumer_key = "aMRKhD08QeYEtA7CcSLcoZiZy"
    consumer_secret = "DEymYahXR6VCoqiRBcqCv0nYFmdtVMxaUN7MXd25Fw0uVOnCHX"

    access_token = "1061925567228928000-Nhyxcap5pwyEeIBu582MbwRqABOYBP"
    access_token_secret = "DOLaUA33vEOsJlXh5dS0167c6jXY2Zu5m8vqUA2PLMVts"

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    # returning API handle
    return tweepy.API(auth, wait_on_rate_limit=True)

if __name__ == '__main__':
    api = setup()
    abstract = raw_input()

    fle=open('temp.txt','w')
    fle.write(abstract)
    fle.close()
    tweet=None
    with open("args.pickle", "rb") as f:
        args = pickle.load(f)

    word_dict, reversed_dict, article_max_len, summary_max_len = build_dict("valid", args.toy)
    valid_x = build_dataset("valid", word_dict, article_max_len, summary_max_len, args.toy)
    valid_x_len = [len([y for y in x if y != 0]) for x in valid_x]

    with tf.Session() as sess:
        model = Model(reversed_dict, article_max_len, summary_max_len, args, forward_only=True)
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state("./saved_model/")
        saver.restore(sess, ckpt.model_checkpoint_path)

        batches = batch_iter(valid_x, [0] * len(valid_x), args.batch_size, 1)

        for batch_x, _ in batches:
            batch_x_len = [len([y for y in x if y != 0]) for x in batch_x]

            valid_feed_dict = {
                model.batch_size: len(batch_x),
                model.X: batch_x,
                model.X_len: batch_x_len,
            }

            prediction = sess.run(model.prediction, feed_dict=valid_feed_dict)
            prediction_output = [[reversed_dict[y] for y in x] for x in prediction[:, 0, :]]

            with open("result.txt", "a") as f:
                for line in prediction_output:
                    summary = list()
                    for word in line:
                        if word == "</s>":
                            break
                        if word not in summary:
                            summary.append(word)
                    sum=""
                    for word in summary:
                        if(word!="#"):
                            sum=sum+(word.encode("utf-8")+ " ")
                            sum.decode("utf-8").encode("utf-8")
                    tweet=sum

    #print tweet
    # load the model here and use abstract to generate tweet
    # in `tweet` variable

    api.update_status(tweet)
    #print 'generated tweet ', tweet
    print 'Tweeted on twitter!'
