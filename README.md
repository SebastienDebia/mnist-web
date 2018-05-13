# mnist-web
This little snippet recognises handwritten digits. 
It was made as a support for teaching neural networks, as it is more fun to play with than accuracy number on the test set!

The input is done by the user in a web page using javascript. The model runs with tensorflow (1.5) and is served using flask.

2 models are provided:
 - one very simple to demonstrate tensorflow's lower level API
 - a more complex CNN
 
The second model is used by default.
It is loosely inspired by mobilenet and LeCun's models.

I added distortions to the input images, which reduced a little bit the accuracy but increased a lot the result with actual user input which is often more ... warped, rotated, zommed in or out .. than the original mnist.
I added entropy to the loss, to try and make the network less confident. It seemed to have a positive impact on the learning.

Overall the accuracy reaches ~99.3%

# demo
![](https://user-images.githubusercontent.com/5497622/39972447-3f8468dc-5710-11e8-91bf-0e674be394a8.gif)
