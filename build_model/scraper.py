import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
from datetime import datetime
import logging
import time
import io
from urllib.parse import urlparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('web_scraper')

class WebScraper:
    def __init__(self):
        """Initialize the web scraper with default headers to mimic a browser request."""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'Referer': 'https://www.google.com/'  # Adding referer to avoid some anti-scraping measures
        }
        # Common patterns for navigation, sidebar, and footer elements to exclude
        self.exclude_patterns = [
            'nav', 'navbar', 'menu', 'sidebar', 'footer', 'header', 'breadcrumb',
            'pagination', 'advertisement', 'ad-', 'banner', 'cookie', 'popup',
            'navigation', 'social', 'search', 'meta', 'related', 'recommended',
            'share', 'comment', 'widget', 'toolbar', 'dialog'
        ]
        
        # NEW: Data table indicators - positive signals for detecting real data tables
        self.data_table_indicators = [
            'data', 'stats', 'statistics', 'results', 'rankings', 'standings',
            'list', 'summary', 'comparison', 'values', 'metrics', 'report',
            'country', 'year', 'date', 'price', 'rate', 'score', 'rank'
        ]
        
    def get_page_content(self, url):
        """
        Fetch the content of a web page with retry logic.
        
        Args:
            url (str): The URL to fetch
            
        Returns:
            BeautifulSoup object or None if fetching fails
        """
        # Validate URL format
        if not re.match(r'^https?://', url):
            url = 'https://' + url
            
        # Get the domain to set as referer for some sites that block scraping
        parsed_url = urlparse(url)
        domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        self.headers['Referer'] = domain
            
        # Try up to 3 times with increasing delays
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt+1} to fetch URL: {url}")
                
                # Add a delay between retries to avoid being flagged as a bot
                if attempt > 0:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                
                response = requests.get(url, headers=self.headers, timeout=30)
                
                # Handle common HTTP errors
                if response.status_code == 403:
                    logger.warning(f"Access forbidden (403) for URL: {url}")
                    # Try with a different user agent on next attempt
                    self.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:98.0) Gecko/20100101 Firefox/98.0'
                    continue
                    
                response.raise_for_status()  # Raise exception for other 4XX/5XX responses
                
                # Check if the response is HTML
                content_type = response.headers.get('Content-Type', '')
                if not ('text/html' in content_type or 'application/xhtml+xml' in content_type):
                    logger.warning(f"URL does not return HTML content: {url}")
                    return None
                    
                # Try to detect the encoding
                if 'charset' in content_type:
                    encoding = re.search(r'charset=([^\s;]+)', content_type).group(1)
                else:
                    encoding = response.apparent_encoding
                
                # Set encoding for proper character handling
                response.encoding = encoding
                
                return BeautifulSoup(response.text, 'html.parser')
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching URL {url} (attempt {attempt+1}): {str(e)}")
                if attempt == max_retries - 1:
                    return None  # Return None after last retry
                    
        return None
    
    def _should_exclude_element(self, element):
        """
        Check if an element should be excluded based on class or id attributes.
        
        Args:
            element: BeautifulSoup element to check
            
        Returns:
            bool: True if the element should be excluded
        """
        if not element:
            return False
            
        # Check class and id attributes
        element_class = element.get('class', [])
        element_id = element.get('id', '')
        
        # Convert class list to string for easier checking
        if isinstance(element_class, list):
            element_class = ' '.join(element_class)
            
        # Check against exclude patterns
        for pattern in self.exclude_patterns:
            if (pattern in element_class.lower() or 
                pattern in element_id.lower()):
                return True
                
        # Check parent elements (only 2 levels now - was 3)
        parent = element.parent
        for _ in range(2):
            if parent is None:
                break
                
            parent_class = parent.get('class', [])
            parent_id = parent.get('id', '')
            
            if isinstance(parent_class, list):
                parent_class = ' '.join(parent_class)
                
            for pattern in self.exclude_patterns:
                if (pattern in parent_class.lower() or 
                    pattern in parent_id.lower()):
                    return True
                    
            parent = parent.parent
            
        return False
    
    def _is_data_table(self, table):
        """
        NEW METHOD: Check if a table is likely a data table rather than a layout table.
        
        Args:
            table: BeautifulSoup table element
            
        Returns:
            bool: True if the table appears to contain data
        """
        # Check 1: Look for positive indicators in table attributes
        table_class = table.get('class', [])
        table_id = table.get('id', '')
        
        if isinstance(table_class, list):
            table_class = ' '.join(table_class)
            
        # Look for positive indicators in class/id
        for indicator in self.data_table_indicators:
            if (indicator in table_class.lower() or indicator in table_id.lower()):
                return True
                
        # Check 2: Count cells with numerical content
        cells = table.find_all(['td', 'th'])
        if not cells or len(cells) < 4:  # Table too small to be meaningful
            return False
            
        # Count cells with numerical content
        numeric_pattern = re.compile(r'^\s*[\d,\.]+%?\s*$')
        numeric_cells = sum(1 for cell in cells if cell.text and numeric_pattern.match(cell.text.strip()))
        
        # If more than 20% of cells have numerical content, likely a data table
        if numeric_cells / len(cells) > 0.2:
            return True
            
        # Check 3: Check if table has header cells
        if table.find_all('th'):
            return True
            
        # Check 4: Check if table has multiple rows with similar structure
        rows = table.find_all('tr')
        if len(rows) >= 3:
            # Check if first few rows have similar number of cells
            cell_counts = [len(row.find_all(['td', 'th'])) for row in rows[:min(5, len(rows))]]
            if len(set(cell_counts)) <= 2 and min(cell_counts) >= 2:
                return True
                
        # Check 5: Look for common data patterns in cell contents
        data_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # dates
            r'\$[\d,.]+',          # currency
            r'\d+\.\d+',           # decimals
            r'\d{1,3}(,\d{3})+',   # formatted numbers
            r'\d+%'                # percentages
        ]
        
        for pattern in data_patterns:
            regexp = re.compile(pattern)
            matches = sum(1 for cell in cells if cell.text and regexp.search(cell.text))
            if matches >= 3:  # At least 3 matches for a pattern
                return True
                
        return False
    
    def extract_tables(self, soup):
        """
        Extract meaningful HTML tables from the soup object,
        filtering out navigation/UI tables.
        
        Args:
            soup (BeautifulSoup): The parsed HTML content
            
        Returns:
            list: List of pandas DataFrames, each representing a table
        """
        if not soup:
            return []
            
        tables = []
        # Find all table elements
        html_tables = soup.find_all('table')
        
        logger.info(f"Found {len(html_tables)} table elements in HTML")
        
        for i, table in enumerate(html_tables):
            # Skip tables that are likely navigation or UI elements
            if self._should_exclude_element(table):
                logger.info(f"Skipping table {i+1} - appears to be navigation/UI element")
                continue
            
            # NEW: Check if the table is likely a data table
            if not self._is_data_table(table):
                logger.info(f"Skipping table {i+1} - does not appear to be a data table")
                continue
                
            # Skip very small tables (likely UI elements)
            rows = table.find_all('tr')
            if len(rows) < 2:  # Changed from 3 to 2
                logger.info(f"Skipping table {i+1} - too few rows ({len(rows)})")
                continue
                
            # Check if table has good data structure with proper th or td cells
            if len(table.find_all(['th', 'td'])) < 4:  # Changed from 5 to 4
                logger.info(f"Skipping table {i+1} - not enough data cells")
                continue
                
            try:
                # Handle tables with specific parsing requirements
                table_html = str(table)
                
                # Try with different parsers and options
                parser_options = [
                    {'parser': 'html5lib'},
                    {'parser': 'lxml', 'flavor': 'html5lib'},
                    {'parser': 'lxml', 'flavor': 'bs4'},
                    {'parser': None}  # Let pandas choose
                ]
                
                success = False
                for options in parser_options:
                    try:
                        # Use StringIO to avoid FutureWarning
                        if 'parser' in options and options['parser']:
                            if 'flavor' in options:
                                dfs = pd.read_html(io.StringIO(table_html), 
                                                  flavor=options['flavor'], 
                                                  parser=options['parser'])
                            else:
                                dfs = pd.read_html(io.StringIO(table_html), 
                                                  parser=options['parser'])
                        else:
                            dfs = pd.read_html(io.StringIO(table_html))
                            
                        success = True
                        break
                    except Exception as e:
                        logger.debug(f"Parser option failed: {options} - {str(e)}")
                
                if not success:
                    # If all parsing methods failed, try a manual approach
                    df = self._manually_parse_table(table)
                    if df is not None:
                        dfs = [df]
                    else:
                        logger.warning(f"All parsers failed for table {i+1}")
                        continue
                
                for df in dfs:
                    # Skip tables with very few columns
                    if df.shape[1] < 2:
                        continue
                        
                    # Clean column names - replace spaces with underscores and remove special characters
                    df.columns = [re.sub(r'[^\w\s]', '', str(col)).strip().replace(' ', '_').lower() for col in df.columns]
                    
                    # Remove completely empty rows and columns
                    df.dropna(how='all', inplace=True)
                    df.dropna(axis=1, how='all', inplace=True)
                    
                    # Skip tables with too few rows or columns after cleaning
                    if df.empty or df.shape[0] < 2 or df.shape[1] < 2:  # Changed from 3 to 2
                        continue
                        
                    # NEW: Better link detection for navigation tables
                    # Filter out tables that are likely navigation elements (lots of URLs or link text)
                    link_pattern = re.compile(r'http|www|\[edit\]|\[citation needed\]')
                    link_count = df.astype(str).apply(lambda x: x.str.contains(link_pattern).sum()).sum()
                    
                    # If more than 70% of cells contain links, skip this table (changed from 50%)
                    if link_count > (df.shape[0] * df.shape[1] * 0.7):
                        logger.info(f"Skipping table {i+1} - likely a navigation table with many links")
                        continue
                    
                    # NEW: Check data quality
                    # Tables with too many identical cells are likely not data tables
                    if df.shape[0] > 1:
                        # Get the most common value in each column and its frequency
                        most_common_counts = df.apply(lambda x: x.value_counts().iloc[0] if not x.empty else 0)
                        # If any column has the same value in >90% of rows, likely not a data table
                        if any(most_common_counts > df.shape[0] * 0.9):
                            # Skip unless it's a small table with few rows
                            if df.shape[0] > 5:
                                logger.info(f"Skipping table {i+1} - columns with too many identical values")
                                continue
                    
                    # Table passed all filters, so add it to results
                    tables.append(df)
                    logger.info(f"Found table {i+1} with shape {df.shape}")
            except Exception as e:
                logger.error(f"Error parsing table {i+1}: {str(e)}")
                
        return tables
    
    def _manually_parse_table(self, table):
        """
        Manually parse an HTML table as a fallback when pandas parsers fail.
        
        Args:
            table: BeautifulSoup table element
            
        Returns:
            pandas DataFrame or None if parsing fails
        """
        try:
            # Get headers
            headers = []
            header_row = table.find('thead')
            if header_row:
                headers = [th.get_text().strip() for th in header_row.find_all('th')]
            
            # If no headers found, try first row as header
            if not headers:
                first_row = table.find('tr')
                if first_row:
                    headers = [th.get_text().strip() for th in first_row.find_all(['th', 'td'])]
            
            # If still no headers, use generic column names
            if not headers:
                # Try to determine number of columns from first data row
                first_data_row = table.find('tr')
                if first_data_row:
                    num_cols = len(first_data_row.find_all(['td', 'th']))
                    headers = [f"Column_{i+1}" for i in range(num_cols)]
                else:
                    return None
            
            # Extract data rows
            data = []
            for row in table.find_all('tr')[1:] if headers else table.find_all('tr'):
                # Skip header row if we already extracted headers
                if row == table.find('tr') and not headers:
                    continue
                    
                row_data = [cell.get_text().strip() for cell in row.find_all(['td', 'th'])]
                
                # Skip empty rows
                if not row_data or all(not cell for cell in row_data):
                    continue
                    
                # Pad or truncate row data to match header length
                if len(row_data) < len(headers):
                    row_data.extend([''] * (len(headers) - len(row_data)))
                elif len(row_data) > len(headers):
                    row_data = row_data[:len(headers)]
                    
                data.append(row_data)
            
            # Create DataFrame
            if data:
                return pd.DataFrame(data, columns=headers)
                
            return None
        except Exception as e:
            logger.error(f"Manual table parsing failed: {str(e)}")
            return None
    
    def find_main_content_area(self, soup):
        """
        Try to identify the main content area of the page.
        
        Args:
            soup (BeautifulSoup): The parsed HTML content
            
        Returns:
            BeautifulSoup element: Main content area or original soup if not found
        """
        if not soup:
            return soup
            
        # Look for common main content identifiers
        main_selectors = [
            'main', 'article', '#content', '#main', '.content', '.main', 
            '[role=main]', '.main-content', '#mainContent', '.post-content',
            '.entry-content', '#primary', '.primary', '.wikitable', '#bodyContent',
            '.mw-content-ltr', '#mw-content-text'  # Added Wikipedia-specific selectors
        ]
        
        for selector in main_selectors:
            try:
                if selector.startswith('#'):
                    element = soup.find(id=selector[1:])
                elif selector.startswith('.'):
                    element = soup.find(class_=selector[1:])
                elif selector.startswith('['):
                    attr, value = selector[1:-1].split('=')
                    element = soup.find(attrs={attr: value})
                else:
                    element = soup.find(selector)
                    
                if element and len(element.get_text(strip=True)) > 500:
                    return element
            except Exception:
                continue
                
        # If no main area found, return original soup
        return soup
    
    def extract_dynamic_table(self, url):
        """
        Special handling for sites with dynamic content loaded via JavaScript.
        Uses selenium to render the page if available.
        
        Args:
            url (str): The URL to scrape
            
        Returns:
            list: List of pandas DataFrames
        """
        try:
            # Import selenium only if needed (not required for installation)
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            import time
            
            logger.info("Using Selenium to handle dynamic content")
            
            # Setup headless Chrome
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-infobars")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument(f"user-agent={self.headers['User-Agent']}")
            
            # Initialize the driver
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=chrome_options
            )
            
            # Load the page
            driver.get(url)
            time.sleep(5)  # Wait for JavaScript to execute
            
            # Get the page source after JavaScript execution
            html_content = driver.page_source
            driver.quit()
            
            # Parse the HTML and extract tables
            soup = BeautifulSoup(html_content, 'html.parser')
            return self.extract_tables(soup)
            
        except ImportError:
            logger.warning("Selenium not available. Cannot handle dynamic content.")
            return []
        except Exception as e:
            logger.error(f"Error extracting dynamic content: {str(e)}")
            return []
    
    def scrape_url(self, url):
        """
        Main function to scrape tables from a URL
        
        Args:
            url (str): The URL to scrape
            
        Returns:
            tuple: (list of DataFrames, success status)
        """
        logger.info(f"Starting to scrape URL: {url}")
        
        # Special handling for known dynamic content sites
        dynamic_content_sites = [
            'nba.com', 'espn.com', 'twitter.com', 'facebook.com',
            'linkedin.com', 'instagram.com', 'coinmarketcap.com',
            'yahoo.finance.com', 'tradingview.com', 'basketball-reference.com',
            'spotrac.com', 'worldometers.info'  # Added more sites
        ]
        
        is_dynamic = any(site in url.lower() for site in dynamic_content_sites)
        
        tables = []
        
        # First try normal scraping
        soup = self.get_page_content(url)
        if soup:
            # NEW: Special handling for Wikipedia
            if 'wikipedia.org' in url.lower():
                # For Wikipedia, specifically look for tables with class 'wikitable'
                wiki_tables = soup.find_all('table', class_='wikitable')
                if wiki_tables:
                    logger.info(f"Found {len(wiki_tables)} wikitables on Wikipedia page")
                    for i, table in enumerate(wiki_tables):
                        try:
                            df = pd.read_html(io.StringIO(str(table)))[0]
                            # Clean column names
                            df.columns = [re.sub(r'[^\w\s]', '', str(col)).strip().replace(' ', '_').lower() for col in df.columns]
                            df.dropna(how='all', inplace=True)
                            df.dropna(axis=1, how='all', inplace=True)
                            tables.append(df)
                            logger.info(f"Found Wikipedia table {i+1} with shape {df.shape}")
                        except Exception as e:
                            logger.error(f"Error parsing Wikipedia table {i+1}: {str(e)}")
            
            # If no tables found with Wikipedia-specific approach, try normal approach
            if not tables:
                # Try to identify main content area to focus scraping
                main_content = self.find_main_content_area(soup)
                
                # Extract data from tables
                tables = self.extract_tables(main_content)
                
                if not tables:
                    # If no tables found in main content, try with full page
                    if main_content != soup:
                        logger.info("No tables found in main content area, trying full page")
                        tables = self.extract_tables(soup)
        
        # If no tables found and site is likely dynamic, try with selenium
        if not tables and is_dynamic:
            logger.info("No tables found with standard scraping. Site may have dynamic content.")
            try:
                tables = self.extract_dynamic_table(url)
            except Exception as e:
                logger.error(f"Dynamic content extraction failed: {str(e)}")
        
        # NEW: Special handling for Yahoo Finance
        if not tables and 'finance.yahoo.com' in url.lower():
            logger.info("Attempting specialized Yahoo Finance handling")
            try:
                # Look for specific data patterns
                price_tables = soup.find_all('table', {'data-test': 'historical-prices'})
                if price_tables:
                    for table in price_tables:
                        try:
                            df = pd.read_html(io.StringIO(str(table)))[0]
                            tables.append(df)
                            logger.info(f"Found Yahoo Finance table with shape {df.shape}")
                        except Exception as e:
                            logger.error(f"Error parsing Yahoo Finance table: {str(e)}")
                            
            except Exception as e:
                logger.error(f"Yahoo Finance handling failed: {str(e)}")
                
        # NEW: Special handling for CSS tables (display: table)
        if not tables and soup:
            logger.info("Looking for CSS tables (display: table)")
            try:
                # Look for divs that might be acting as tables
                potential_tables = soup.find_all(['div', 'section'], class_=lambda c: c and ('table' in c.lower() or 'grid' in c.lower()))
                
                for i, pt in enumerate(potential_tables):
                    if len(pt.find_all(['div', 'span'])) > 10:  # Must have at least some elements
                        try:
                            # Try to manually extract structured data
                            rows = pt.find_all(class_=lambda c: c and ('row' in c.lower() or 'tr' in c.lower()))
                            
                            if len(rows) >= 2:  # Need at least a header and data row
                                # Extract header
                                header_row = rows[0]
                                header_cells = header_row.find_all(class_=lambda c: c and ('cell' in c.lower() or 'col' in c.lower() or 'td' in c.lower()))
                                headers = [cell.get_text().strip() for cell in header_cells]
                                
                                if not headers or len(headers) < 2:
                                    continue
                                    
                                # Extract data
                                data = []
                                for row in rows[1:]:
                                    cells = row.find_all(class_=lambda c: c and ('cell' in c.lower() or 'col' in c.lower() or 'td' in c.lower()))
                                    row_data = [cell.get_text().strip() for cell in cells]
                                    
                                    if row_data and len(row_data) >= 2:  # Skip rows with too few cells
                                        # Pad or truncate to match header length
                                        if len(row_data) < len(headers):
                                            row_data.extend([''] * (len(headers) - len(row_data)))
                                        elif len(row_data) > len(headers):
                                            row_data = row_data[:len(headers)]
                                        data.append(row_data)
                                
                                if data:
                                    df = pd.DataFrame(data, columns=headers)
                                    # Clean column names
                                    df.columns = [re.sub(r'[^\w\s]', '', str(col)).strip().replace(' ', '_').lower() for col in df.columns]
                                    df.dropna(how='all', inplace=True)
                                    df.dropna(axis=1, how='all', inplace=True)
                                    
                                    if not df.empty and df.shape[0] >= 2 and df.shape[1] >= 2:
                                        tables.append(df)
                                        logger.info(f"Found CSS table {i+1} with shape {df.shape}")
                        except Exception as e:
                            logger.error(f"Error parsing CSS table {i+1}: {str(e)}")
            except Exception as e:
                logger.error(f"CSS table extraction failed: {str(e)}")
        
        if not tables:
            logger.warning("No tables found on the page")
            return [], False
            
        logger.info(f"Successfully scraped {len(tables)} tables from {url}")
        return tables, True
    
    def save_to_excel(self, dataframes, url, output_dir=None):
        """
        Save the extracted DataFrames to Excel files
        
        Args:
            dataframes (list): List of pandas DataFrames
            url (str): The URL that was scraped
            output_dir (str): Directory to save the Excel file (default: current directory)
            
        Returns:
            str: Path to the saved Excel file or None if save failed
        """
        if not dataframes:
            logger.warning("No data to save")
            return None
            
        # Create a safe filename from the URL
        domain = re.sub(r'^https?://', '', url)
        domain = re.sub(r'[^\w.-]', '_', domain)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scraped_{domain}_{timestamp}.xlsx"
        
        # Set output directory
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        file_path = os.path.join(output_dir or "", filename)
        
        try:
            # Create Excel writer
            with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                for i, df in enumerate(dataframes):
                    # Try to create a more meaningful sheet name based on the first column header
                    if df.columns[0] and len(str(df.columns[0])) > 3:
                        sheet_name = f"{str(df.columns[0])[:20]}"
                        sheet_name = re.sub(r'[^\w\s]', '', sheet_name).strip()
                    else:
                        sheet_name = f"Table_{i+1}"
                    
                    # Ensure sheet name length is valid for Excel
                    if len(sheet_name) > 31:
                        sheet_name = sheet_name[:31]
                    
                    # Ensure sheet name is unique
                    if sheet_name in writer.sheets:
                        sheet_name = f"{sheet_name}_{i+1}"
                        if len(sheet_name) > 31:
                            sheet_name = sheet_name[:31]
                    
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Auto-adjust columns width
                    worksheet = writer.sheets[sheet_name]
                    for j, col in enumerate(df.columns):
                        # Get maximum length of data in column
                        column_width = max(
                            df[col].astype(str).apply(len).max(),
                            len(str(col))
                        ) + 2  # Add a little extra space
                        
                        # Excel has a column width limit
                        column_width = min(column_width, 50)
                        
                        # Set column width
                        worksheet.set_column(j, j, column_width)
            
            logger.info(f"Data successfully saved to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving Excel file: {str(e)}")
            return None

# def check_required_libraries():
#     """
#     Check if required libraries are installed and install them if missing.
#     Returns True if all requirements are met, False otherwise.
#     """
#     required_libraries = {
#         'pandas': 'pandas',
#         'bs4': 'beautifulsoup4',
#         'requests': 'requests',
#         'xlsxwriter': 'xlsxwriter',
#         'html5lib': 'html5lib',
#         'lxml': 'lxml'
#     }
    
#     missing_libraries = []
    
#     for module, package in required_libraries.items():
#         try:
#             __import__(module)
#         except ImportError:
#             missing_libraries.append(package)
    
#     if missing_libraries:
#         print(f"Missing required libraries: {', '.join(missing_libraries)}")
#         print("Installing missing libraries...")
        
#         try:
#             import subprocess
#             for package in missing_libraries:
#                 subprocess.check_call(['pip', 'install', package])
#             print("All required libraries installed successfully!")
#             return True
#         except Exception as e:
#             print(f"Error installing libraries: {str(e)}")
#             print("Please install the following packages manually:")
#             for package in missing_libraries:
#                 print(f"  pip install {package}")
#             return False
    
#     return True

# def scrape_and_save(url, output_dir=None):
#     """
#     Function to handle the complete process of scraping a URL and saving to Excel
    
#     Args:
#         url (str): URL to scrape
#         output_dir (str): Directory to save the Excel file
        
#     Returns:
#         tuple: (path to Excel file or None, success status)
#     """
#     # Check if required libraries are installed
#     if not check_required_libraries():
#         return None, False
    
#     scraper = WebScraper()
#     dataframes, success = scraper.scrape_url(url)
    
#     if not success:
#         return None, False
        
#     file_path = scraper.save_to_excel(dataframes, url, output_dir)
#     return file_path, file_path is not None

# # Example usage
# if __name__ == "__main__":
#     # List of example URLs with tables that should work well
#     example_urls = [
#         "https://www.basketball-reference.com/leagues/NBA_2023_per_game.html",
#         "https://finance.yahoo.com/quote/AAPL/history/",
#         "https://en.wikipedia.org/wiki/List_of_countries_by_population",
#         "https://finance.yahoo.com/cryptocurrencies",
#         "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)",
#         "https://en.wikipedia.org/wiki/List_of_chemical_elements",
#     ]
    
#     print("Web Scraper Example")
#     print("===================")
#     print("This script scrapes tables from a URL and saves them to Excel.")
#     print("\nExample URLs to try:")
#     for i, url in enumerate(example_urls):
#         print(f"{i+1}. {url}")
    
#     user_url = input("\nEnter a URL to scrape (or choose a number from examples): ")
    
#     # Check if user entered a number from examples
#     if user_url.isdigit() and 1 <= int(user_url) <= len(example_urls):
#         url_to_scrape = example_urls[int(user_url) - 1]
#     else:
#         url_to_scrape = user_url
    
#     print(f"\nScraping {url_to_scrape}...")
#     file_path, success = scrape_and_save(url_to_scrape)
    
#     if success:
#         print(f"\nSuccess! Data saved to: {file_path}")
#     else:
#         print("\nFailed to scrape data from the URL. Please try a different URL.")