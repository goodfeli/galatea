from galatea.datasets.norb_tiny import NORB_Tiny

train = NORB_Tiny('train')

from pylearn2.datasets.preprocessing import Pipeline, GlobalContrastNormalization, ZCA

pipeline = Pipeline()

pipeline.items = [ GlobalContrastNormalization(), ZCA() ]

train.apply_preprocessor(pipeline, can_fit = True)

from pylearn2.utils.serial import save

save('norb_tiny_preprocessed_train.pkl', train)
save('norb_tiny_preprocessor.pkl', pipeline)

