import scrapy


class InfoSpider(scrapy.Spider):
    name = "info"

    def start_requests(self):
        urls = [
            'http://www.info.sk/spravy/{}/'.format(i) for i in range(1,9603)

        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split('/')[-2]
        self.logger.info('Page  {}'.format(page))
        for a in response.css('#content div a'):
            if a.css('h2').extract_first() is not None:
                href = a.css('::attr(href)').extract_first()
                yield response.follow(href, self.parse_artickle)

    def parse_artickle(self, response):
        html = response.css('#sitelement').extract_first()
        yield {
            'url': response.url,
            'html': html
        }
