# About
The undertaking of this project was motivated by my interest in working the Animal Crossing domain, which I am very fond of.
The Animal Crossing franchise is one I have enjoyed since 2006, and sequence modelling has provided an avenue to analyse one
of its mechanics (Stalk Market), whilst also furthering my understanding of the topic itself. I hope this body of work motivates
further efforts to determine the efficacy of sequence modelling in other artificial game market domains, which has not been studied
frequently.

Sequence modelling has seen fruitful use in various applications such as speech recognition, music generation, sentiment
classification, machine translation, and time series prediction. However, an area that is not widely studied is the upcoming time
series classification variant seen in artificial game markets. With various potential candidate artificial game markets to analyse, I
was interested in implementing a time series classification for a personal favourite - Nintendo’s Animal Crossing: New Horizons
(ACNH).
In hopes of exploiting the benefits of sequence modelling, this project aims to determine the efficacy of various machine learning
and sequence modelling techniques in classifying an island’s turnip prices to its correct weekly trend. Additionally, I hope to
provide this reliable trend classification as early in the week as possible.
This approach will come in four broad phases: data acquisition and pre-processing, exploratory data analysis, clustering analysis,
and model evaluations.

# Background
Many games feature an in-game trading space, both online and offline. Games such as Counterstrike: Global Offensive, Eve Online,
Warframe, and Elder Scrolls Online instantly spring to mind, but there are tens or potentially hundreds of other games that feature
artificial market spaces. Along with an online and offline aspect, game markets have two main approaches when handling currency:

1. Real-life money: USD is the most commonly adopted currency with games like CS: GO and
2. Artificial game-specific tokens/currency: For example, Warframe uses an artificial currency called platinum for its trading.

With various potential candidate artificial game markets to analyse, I am interested in implementing a time series classification for
a personal favourite of mine: Nintendo’s Animal Crossing: New Horizons (ACNH). This is an offline game featuring an artificial
in-game token called Bells. Unlike the previous game titles referenced above, Animal Crossing is an offline game, thus removing
the negative public conception of online trading in video games. It is a light-hearted and child-friendly title that features an in-game
market space where the player interacts with the game AI instead of real-life players.

![Animal-Crossing-New-Horizons-Turnip-Prices-Guide-What-to-Buy-and-Sell-For](https://user-images.githubusercontent.com/71750671/183031356-fc07d6fd-90b2-4f97-b7c1-19ad0ae08420.jpg)

The in-game market is aptly named the Stalk Market, as players trade the in-game currency, Bells, in exchange for turnips. The
Stalk Market follows quite simple mechanics: on Sunday mornings, a turnip vendor named Daisy Mae will visit the player’s island,
where she will offer to sell her turnips at a randomly generated price. Then during the subsequent days of the week (Monday to
Saturday), the island shop will offer to buy the turnips for a price, in Bells, that changes twice daily. The aim is to maximise profits
by buying low and selling high.

The strategy comes in determining how the turnip prices will unfold (i.e., the trend) as the week progresses. In this project, we will
be exploring the weekly trends that the prices follow, with the objective of being able to classify which trend the turnip prices follow
based on the first few days of pricing data. We explore classifying variable sequence lengths indicative of how a player would
navigate through the week, noting the turnip prices. This trend identification can allow the player to determine when to sell their
turnips and thus conquer the Stalk Market.

# Results and Conclusion Summary
From the results presented in the report, the following main observations were made:

1. The LSTM and GRU provide more accurate classifications than the Naïve Bayes classifier.
2. The classification accuracy is directly proportional to the number of provided turnip prices during the week.
3. As the number of provided turnip decrease, the accuracy difference between the naïve classifier and the RNN models
increase.
4. The GRU is the best performing classifier.

This research has proven that sequence modelling can be a viable strategy to employ in time series classification of artificial game
markets. Not only is sequence modelling a viable machine learning technique in this domain, but as empirically shown, it performs
better than the probabilistic model baseline representing more conventional approaches. These gains are especially welcomed in
ACNH’s market, where we try to make classifications as soon as possible with limited input data, which saw desirable results
achieved through the various sequence lengths. The extended goal of being able to provide some kind of forecasting ability was
achieved through the more descriptive clustering analysis. This allowed the classification class itself to indicate which day and time
one could expect a price spike. This project was conducted in an empirical manner, creating comparisons for two clustering
techniques and three classification models, which concluded with DBSCAN and GRU being the best respective pairing for a
sequence modelling-based time series classification task, whilst indicating some predictive power.
