License: [WTFPL](https://en.wikipedia.org/wiki/WTFPL)

Visualization of what it's doing right now.
https://dl.dropbox.com/u/9632169/brain_demos/stdp_and_ltp.html

This brain drastically simplifies some aspects of Numenta's CLA's and adds complexity to others.
This has been done in the name of achieving greater similarity to biological brains.
For instance, context is not represented by a set of four neurons in a column,
but rather is passed down through feedback connections with higher level layers.
Also, the idea of crowding out neurons to form sparse representations will be replaced by increasing leaf level
surface area and making leaf level neurons more specialized to certain types of input.

Code follows these style rules: http://code.google.com/p/soc/wiki/PythonStyleGuide

Roadmap:
- Make correct one level brain predictions. Done
- Form invariant representations.
  - Show layers.
  - Create invariant representation at top level.
  - Pool input under that representation while input is falls within a reasonable expectation.
  - When unexpected, form a new invariant represenation to pool the new input under with a weak link to previous invariant representation.
  - Increase links between invariant represenations as they co-occur and combine them if links are strong enough.
  - When there is too much unexpected data, avoid more input in some way. (Overexcitation)
    - Some neurons (i.e. color neurons) are more sensitive to change than other neurons. That's why fast moving objects are mostly in black / white.
      - These neurons are adjacent to less sensitive neurons.
- Predict different outcome for same input based on context.
- Introduce multiple inputs. i.e. text and visual and audio and location, etc...
- Condense invariant representation chains (compression/source coding/ decision tree).
- Give mechanism for output / interaction with video say (control stopping, attention, switching, etc…)
- Scale up input to allow phone sized video input.

Two threads
 - Predicting thread writes predictions, reads signals.
   - Transforms abstract prediction to more concrete ones by reading signal from layer below.
 - Signal thread writes signals, reads predictions.
   - Writes signal to layer above, only if unexpected.

 - USE pipes to do this. http://docs.python.org/library/multiprocessing.html#exchanging-objects-between-processes

- Release phone brainy, bitty, bitkid, bitbaby, bitbrain app.
OR
- Release game where brains can compete.

- Train best brains from community on web data.

Idea:
- Use past 5 time cycles to predict next cycle on same level.
  - 2 cycles ago predicts 2 cycles ahead OR 4 older cycles provide inhibitive effect.

- Feed SURF data in...
- ? Map pixels to different locations based on shade or skip brain and just use SVM
- Rather than mush all temporally close data, compress it.


TODO: Maintain 4 dimensional numpy array. (x, y, z, t) of floats for neuron values
      along with 3 dimensional array (x, y, z) for connections
      to allow more numpy numeric calculations.

      Write ethics post on machine learning.
      - Inevitable
      - No desire for survival
      - No happiness, sadness, anger
      - Will be like cheat-mirroring a chess game until better integration.
      - Current societal and biological problems will be replaced with more difficult space/time problems.
      - Connecting to machine brains will be analogous to connecting to the internet.

      Finish HTM's vs DBN's.

      Combine Probablistic Max Pooling with temporal pooling.
      - Read paper on how this was done.
      - Do a difference calculation between frames.
        - Gray is just on.
        - White
        - Are prediction feedbacks doing the same thing?


Thoughts from talking w/ Christina
  - Orientation, time of day, and  location may not be that valuable, compared to difficulty in getting that data.
  - Combining time inference at all levels with novelty based controls and deep learning

Questions:
  Why didn't Google use intralayer connections? - Appears to be convetion in traditional neural nets.
d

