## Here we prepare scrapy spider for our use

# We will create parse to get links from catalogue page
# And parse_book for book data from each book link
import scrapy

class SpideronBook(scrapy.Spider):
    name = "books"
    allowed_domains = ["books.toscrape.com"]
    start_urls = ["https://books.toscrape.com/catalogue/page-1.html"]

    # We only want to scrape 5 pages to get exactly 100 books (20 per page)
    custom_settings = {
        'CLOSESPIDER_PAGECOUNT': 0, # We will handle page limits manually
    }

    def __init__(self, *args, **kwargs):
        super(SpideronBook, self).__init__(*args, **kwargs)
        self.page_count = 1

    def parse(self, response):
        # 1. Find all the book links on the current page
        books = response.css('h3 a::attr(href)').getall()

        # 2. Tell the spider to visit each book link
        for book_url in books:
            yield response.follow(book_url, callback=self.parse_book)

        # 3. Handle Pagination (Clicking the 'Next' button)
        next_page = response.css('li.next a::attr(href)').get()
        if next_page and self.page_count < 5:
            self.page_count += 1
            yield response.follow(next_page, callback=self.parse)

    def parse_book(self, response):
        # This function extracts data from the individual book page

        # This Extracts rating from the class name
        rating_class = response.css('p.star-rating::attr(class)').get()
        rating = rating_class.replace('star-rating ', '') if rating_class else 'None'

        # This Extracts availability and cleaning up the text
        availability = response.css('p.instock.availability::text').getall()
        availability = "".join(availability).strip()

        yield {
            'title': response.css('h1::text').get(),
            'category': response.css('ul.breadcrumb li:nth-child(3) a::text').get(),
            'price': response.css('p.price_color::text').get(),
            'rating': rating,
            'availability': availability,
            # Description is the paragraph right after the product_description div
            'product_description': response.xpath('//div[@id="product_description"]/following-sibling::p/text()').get(),
            # Information table extraction
            'upc': response.xpath('//th[text()="UPC"]/following-sibling::td/text()').get(),
            'number_of_reviews': response.xpath('//th[text()="Number of reviews"]/following-sibling::td/text()').get(),
            'product_url': response.url
        }
