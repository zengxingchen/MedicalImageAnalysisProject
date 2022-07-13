# ============== 参数设置 =============== #
train_val_data = features[:700, :]
train_val_label = labels[:700]
test_data = features[700:, :]
test_label = labels[700:]

train_data = train_val_data
train_label = train_val_label
val_data = test_data
val_label = test_label