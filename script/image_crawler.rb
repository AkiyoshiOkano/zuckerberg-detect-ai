require "pry"
require "open-uri"
require "FileUtils"
require 'net/http'
require 'json'

# 保存先ディレクトリ
@dirName = "./gates_images/"
# ディレクトリを作る
FileUtils.mkdir_p(@dirName) unless FileTest.exist?(@dirName)

# 引数の画像URLから指定フォルダに画像を保存する
def save_image(url, num)
  filePath = "#{@dirName}/gates#{num.to_s}.jpg"
  open(filePath, 'wb') do |output|
    open(url) do |data|
      output.write(data.read)
    end
  end
end

# Bing Search API(API公式のコードをそのまま使用)
count = 150
search_word = 'bill gates'
uri = URI('https://api.cognitive.microsoft.com/bing/v5.0/images/search')
uri.query = URI.encode_www_form({
    'q' => search_word,
    'count' => count,
    'offset' => 150
})
request = Net::HTTP::Post.new(uri.request_uri)
request['Content-Type'] = 'multipart/form-data'
request['Ocp-Apim-Subscription-Key'] = 'APIキー' # Fix Me
request.body = "{body}"
response = Net::HTTP.start(uri.host, uri.port, :use_ssl => uri.scheme == 'https') do |http|
  http.request(request)
end

# qで指定したワードの画像をcountの数だけ保存する
count.times do |i|
  begin
    image_url = JSON.parse(response.body)["value"][i]["thumbnailUrl"]
    save_image(image_url, i + 150)
  rescue => e
    puts "image#{i} is error!"
    puts e
  end
end


