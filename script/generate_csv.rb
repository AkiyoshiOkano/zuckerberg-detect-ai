require 'fileutils'

train_data_path = "./data/train/data.txt"
test_data_path = "./data/test/data.txt"

FileUtils.touch(train_data_path) unless FileTest.exist?(train_data_path)
FileUtils.touch(test_data_path) unless FileTest.exist?(test_data_path)


test_zuck_data_paths = Dir.glob("./data/test/zuckerberg/*.jpg")
test_elon_data_paths = Dir.glob("./data/test/elonmusk/*.jpg")
test_gates_data_paths = Dir.glob("./data/test/billgates/*.jpg")
train_zuck_data_paths = Dir.glob("./data/train/zuckerberg/*.jpg")
train_elon_data_paths = Dir.glob("./data/train/elonmusk/*.jpg")
train_gates_data_paths = Dir.glob("./data/train/billgates/*.jpg")


File.open(test_data_path, "w") do |f|
  test_zuck_data_paths.each { |path| f.puts("#{path} 0") }
  test_elon_data_paths.each { |path| f.puts("#{path} 1") }
  test_gates_data_paths.each { |path| f.puts("#{path} 2") }
end
File.open(train_data_path, "w") do |f|
  train_zuck_data_paths.each { |path| f.puts("#{path} 0") }
  train_elon_data_paths.each { |path| f.puts("#{path} 1") }
  train_gates_data_paths.each { |path| f.puts("#{path} 2") }
end