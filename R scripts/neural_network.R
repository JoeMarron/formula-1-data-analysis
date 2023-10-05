tic()
# Flags
FLAGS <- flags(flag_integer('dense_units',         32),
               flag_string('activation_function', 'relu'),
               flag_numeric('learning_rate',       0.001),
               flag_numeric('drop_out',            0.1),
               flag_integer('epochs',              50),
               flag_integer('batch_size',          32),
               flag_numeric('validation_split',    0.2))


# Model (3-Hidden-Layer)
model <- keras_model_sequential() %>% 
        layer_dense(units=FLAGS$dense_units,
                    activation = FLAGS$activation_function,
                    input_shape = ncol(train_x)) %>%
        layer_dropout(rate=FLAGS$drop_out) %>%
        layer_dense(units=FLAGS$dense_units/2,
                    activation = FLAGS$activation_function) %>%
        layer_dropout(rate=FLAGS$drop_out) %>%
        layer_dense(units=FLAGS$dense_units/4,
              activation = FLAGS$activation_function) %>%  
        layer_dropout(rate=FLAGS$drop_out) %>%
        layer_dense(units = 1, activation = "sigmoid")



# Compile
model %>%
  compile(
    loss = "binary_crossentropy", 
    optimizer = optimizer_adam(learning_rate=FLAGS$learning_rate),   # Use adam optimizer
    metrics = c("binary_accuracy")  # (for binary classifier)
  )

# Train the model
history <- model %>%
  fit(
    x = as.matrix(x),
    y = y,
    epochs = FLAGS$epochs,
    batch_size = FLAGS$batch_size,
    validation_split = FLAGS$validation_split)
toc()