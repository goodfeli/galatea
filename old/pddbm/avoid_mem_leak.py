fuck = """!obj:galatea.s3c.s3c_dataset.S3C_Dataset {
                            "raw" : !pkl: &src "${PYLEARN2_DATA_PATH}/stl10/stl10_32x32_whitened/unsupervised.pkl",
                                                    "transformer" : !pkl: &s3c "/u/goodfeli/galatea/pddbm/config/stl/full/layer_1_C1.pkl",
                                                                            }"""

from pylearn2.config import yaml_parse

fuck = yaml_parse.load(fuck)

fuck = fuck.get_batch_design(400)

from pylearn2.utils.serial import save

save('fuck.npy',fuck)
