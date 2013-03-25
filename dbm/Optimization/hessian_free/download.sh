echo "Downloading the MNIST and Curves datasets"
mkdir data; cd data
curl http://www-etud.iro.umontreal.ca/~erhandum/hessian_free/data/mnist.pkl.gz > mnist.pkl.gz
curl http://www-etud.iro.umontreal.ca/~erhandum/hessian_free/data/curves.gz > curves.gz

cd ..
mkdir pretrained_models; cd pretrained_models
echo "Downloading the pre-trained models"
curl http://www-etud.iro.umontreal.ca/~erhandum/hessian_free/pretrained_models/mnist_pretrained_model > mnist_pretrained_model
curl http://www-etud.iro.umontreal.ca/~erhandum/hessian_free/pretrained_models/curves_pretrained_model > curves_pretrained_model

echo "Done"
