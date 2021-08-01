## Goal
Given a first sequence of a sine wave, let the model generate the rest of the wave

<br/>

## 1. keras_univariate_lstm_example.ipynb
Keras LSTM model to learn raw sine waves.

Subtle changes on 'n_steps' and 'hid_dim' alters the fit of the model greatly
-> must find a balance between bias (underfitting) and variance (overfitting)
![[keras_sine_generation.png]]

<br/>

## 2. pytorch_lstm_simpleSeq.ipynb
Always start with a simpler problem.
Pytorch LSTM model to learn simple number sequence of length 3.

|sequence	| label|
|---|---|
|[10, 20, 30] | 40 |
|[20, 30, 40] | 50 |
|[30, 40, 50] | 60 |

model with single LSTM layer + single FC layer seems to <span style="color:red"> underfit</span>;

bigger hid_dim == more accurate the sequence generation
(epoch 1000)
![[pytorch_simpleseq_generation.png]]
![[pytorch_simpleseq_loss.png]]

around hid_dim 200 (left), generation accuracy is at tolerable level
around 500 (right), maybe the model starts to <span style="color:hsl(210, 88%, 63%)"> overfit</span>
![[pytorch_simpleseq_generation2.png]]
![[pytorch_simpleseq_loss2.png]]


-> add one more FC layer to incrase complexity? (**LSTM_2FC**)

Doesn't seem like working well (flats out or shows huge bias).
Maybe another FC layer is an overkill for learning simple sequence data?

<br/>

## 3. pytorch_lstm_sinewave.ipynb
1. Using simple LSTM model, train raw sine wave
(epoch 200, Grid-layout of hid_dim)

prediction is pretty accurate.
![[pytorch_sinewave_generation.png]]
![[pytorch_sinewave_loss.png]]

</br>

2. Now, train sine wave with Gaussian noise 

Training data
![[noisy_sinewave.png]]

less accurate sine wave is generated now
(single exp - epoch 200, hid_dim 50)
![[noisy_sinewave_prediction.png]]

perhaps, more complex model is able to learn more complete sine wave from the noises?
→ Not really. Now, hid_dim 20 learns best
![[noisy_sinewave_prediction2.png]]
![[noisy_sinewave_loss2.png]]

</br>

#### Conclusion
* Find **appropriate model capacity**
	- Changes every time according to the type of data
	- Use grid layout for initial search, then hand tuning
* Train for **enough epochs** to reduce loss as low as possible
* Using SGD gave flat-out result (underfitting perhaps)
-> Changed to **Adam** to make model converge
![[flat2.png]]![[flat1.png]]
optimizer matters!

* Additional hyperparameter tuning can be done in the similar way
(learning rate, regularization, ...)