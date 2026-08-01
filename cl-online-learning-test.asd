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
               :rove)
  :components ((:module "t"
                :components
                ((:file "cl-online-learning"))))
  :perform (test-op (o c)
             (unless (uiop:symbol-call :rove :run c)
               (error "Tests failed."))))
