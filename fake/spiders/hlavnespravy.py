import scrapy


class HlavneSpravySpider(scrapy.Spider):
    name = "hlavnespravy"

    def start_requests(self):
        urls = [
            'https://www.hlavnespravy.sk/page/{}'.format(i) for i in range(1,4226) #4226

        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split('/')[-1]
        self.logger.info('Page  {}'.format(page))
        for article in response.css('.col-xs-8.col-content-sm-8.col-content-lg-8.top6.text-wrapper'):
            a = article.css('a::attr(href)').extract_first()
            if 'hlavnespravy.sk/redir' in a:
                continue
            yield response.follow(a, self.parse_artickle)

    def parse_artickle(self, response):
        post = response.css('.hsp-post').extract_first()
        yield {
            'url': response.url,
            'html': post
        }
