#|
  This file is a part of cl-online-learning project.
|#

(in-package :cl-user)
(defpackage cl-online-learning-test-asd
  (:use :cl :asdf))
(in-package :cl-online-learning-test-asd)

(defsystem cl-online-learning-test
  :author ""
  :license ""
  :depends-on (:cl-online-learning
               :prove)
  :components ((:module "t"
                :components
                ((:test-file "cl-online-learning"))))

  :defsystem-depends-on (:prove-asdf)
  :perform (test-op :after (op c)
                    (funcall (intern #.(string :run-test-system) :prove-asdf) c)
                    (asdf:clear-system c)))
