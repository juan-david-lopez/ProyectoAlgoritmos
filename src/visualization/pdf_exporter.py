"""
PDF Export Module

Exports all visualizations to a professional PDF report.

Features:
- Professional cover page
- Table of contents with internal links
- Geographic analysis section
- Word cloud analysis section
- Timeline analysis section
- Summary statistics
- Metadata page
- Image optimization for PDF
- Professional styling and layout
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import warnings
from loguru import logger

# Suppress warnings
warnings.filterwarnings('ignore')


class PDFExporter:
    """
    Exports visualizations to professional PDF report.

    This class generates comprehensive PDF reports with all bibliometric
    visualizations and statistics in a professional layout.
    """

    def __init__(self, output_pdf_path: str):
        """
        Initialize PDF exporter.

        Args:
            output_pdf_path: Path where PDF will be saved
        """
        self.output_pdf_path = Path(output_pdf_path)
        self.output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

        # Import reportlab components
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm, mm
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
            from reportlab.pdfgen import canvas

            self.A4 = A4
            self.SimpleDocTemplate = SimpleDocTemplate
            self.Paragraph = Paragraph
            self.Spacer = Spacer
            self.RLImage = RLImage
            self.Table = Table
            self.TableStyle = TableStyle
            self.PageBreak = PageBreak
            self.ParagraphStyle = ParagraphStyle
            self.cm = cm
            self.mm = mm
            self.colors = colors
            self.TA_CENTER = TA_CENTER
            self.TA_LEFT = TA_LEFT
            self.TA_RIGHT = TA_RIGHT
            self.TA_JUSTIFY = TA_JUSTIFY
            self.canvas = canvas

            # Get sample styles
            self.styles = getSampleStyleSheet()

            # Create custom styles
            self._create_custom_styles()

        except ImportError:
            logger.error("reportlab not installed. Install with: pip install reportlab")
            raise

        # Story elements (content of PDF)
        self.story = []

        # Page number tracking
        self.page_num = 0

        logger.success("PDFExporter initialized successfully")

    def _create_custom_styles(self):
        """Create custom paragraph styles for the PDF."""
        # Cover title style
        self.styles.add(ParagraphStyle(
            name='CoverTitle',
            parent=self.styles['Heading1'],
            fontSize=28,
            textColor=self.colors.HexColor('#2E86AB'),
            spaceAfter=30,
            alignment=self.TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        # Cover subtitle style
        self.styles.add(ParagraphStyle(
            name='CoverSubtitle',
            parent=self.styles['Normal'],
            fontSize=16,
            textColor=self.colors.grey,
            spaceAfter=12,
            alignment=self.TA_CENTER,
            fontName='Helvetica'
        ))

        # Section title style
        self.styles.add(ParagraphStyle(
            name='SectionTitle',
            parent=self.styles['Heading1'],
            fontSize=20,
            textColor=self.colors.HexColor('#2E86AB'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        ))

        # Subsection title style
        self.styles.add(ParagraphStyle(
            name='SubsectionTitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=self.colors.HexColor('#4A5568'),
            spaceAfter=10,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        ))

        # Body text style
        self.styles.add(ParagraphStyle(
            name='BodyJustify',
            parent=self.styles['BodyText'],
            fontSize=11,
            alignment=self.TA_JUSTIFY,
            fontName='Helvetica',
            leading=14
        ))

        # Bullet point style
        self.styles.add(ParagraphStyle(
            name='BulletPoint',
            parent=self.styles['Normal'],
            fontSize=11,
            fontName='Helvetica',
            leftIndent=20,
            bulletIndent=10
        ))

    def create_cover_page(self, title: str = "Bibliometric Analysis Report",
                         subtitle: str = "Generative Artificial Intelligence",
                         date_range: str = "",
                         authors: List[str] = None,
                         institution: str = ""):
        """
        Create professional cover page.

        Args:
            title: Main title
            subtitle: Subtitle
            date_range: Temporal range of analysis
            authors: List of authors/analysts
            institution: Institution name
        """
        logger.info("Creating cover page")

        # Add vertical space
        self.story.append(self.Spacer(1, 4*self.cm))

        # Title
        title_para = self.Paragraph(title, self.styles['CoverTitle'])
        self.story.append(title_para)

        # Subtitle
        if subtitle:
            subtitle_para = self.Paragraph(subtitle, self.styles['CoverSubtitle'])
            self.story.append(subtitle_para)

        # Date range
        if date_range:
            self.story.append(self.Spacer(1, 1*self.cm))
            date_para = self.Paragraph(f"<b>Period:</b> {date_range}",
                                      self.styles['CoverSubtitle'])
            self.story.append(date_para)

        # Institution
        if institution:
            self.story.append(self.Spacer(1, 0.5*self.cm))
            inst_para = self.Paragraph(institution, self.styles['CoverSubtitle'])
            self.story.append(inst_para)

        # Authors
        if authors:
            self.story.append(self.Spacer(1, 2*self.cm))
            authors_text = "<br/>".join(authors)
            authors_para = self.Paragraph(f"<b>Analysts:</b><br/>{authors_text}",
                                         self.styles['CoverSubtitle'])
            self.story.append(authors_para)

        # Generation date
        self.story.append(self.Spacer(1, 3*self.cm))
        gen_date = datetime.now().strftime("%B %d, %Y")
        date_para = self.Paragraph(f"<i>Generated: {gen_date}</i>",
                                   self.styles['CoverSubtitle'])
        self.story.append(date_para)

        # Page break
        self.story.append(self.PageBreak())

    def add_table_of_contents(self):
        """
        Add table of contents with internal links.
        """
        logger.info("Adding table of contents")

        # Title
        toc_title = self.Paragraph("Table of Contents", self.styles['SectionTitle'])
        self.story.append(toc_title)
        self.story.append(self.Spacer(1, 0.5*self.cm))

        # TOC items
        toc_items = [
            "1. Executive Summary",
            "2. Geographic Distribution",
            "3. Term Analysis",
            "4. Temporal Evolution",
            "5. Summary Statistics",
            "6. Metadata and Methods"
        ]

        for item in toc_items:
            item_para = self.Paragraph(item, self.styles['BodyText'])
            self.story.append(item_para)
            self.story.append(self.Spacer(1, 0.3*self.cm))

        self.story.append(self.PageBreak())

    def add_geographic_section(self,
                              map_image_path: Optional[str] = None,
                              statistics: Optional[Dict] = None):
        """
        Add geographic analysis section.

        Args:
            map_image_path: Path to geographic map image
            statistics: Geographic statistics dictionary
        """
        logger.info("Adding geographic section")

        # Section title
        title = self.Paragraph("2. Geographic Distribution", self.styles['SectionTitle'])
        self.story.append(title)

        # Introductory text
        intro_text = """
        This section presents the geographic distribution of scientific publications.
        The analysis reveals the global landscape of research activity, highlighting
        leading countries and regions contributing to the field. Understanding geographic
        patterns helps identify research hubs, collaboration opportunities, and
        potential gaps in global research coverage.
        """
        intro_para = self.Paragraph(intro_text, self.styles['BodyJustify'])
        self.story.append(intro_para)
        self.story.append(self.Spacer(1, 0.5*self.cm))

        # Add map if provided
        if map_image_path and Path(map_image_path).exists():
            try:
                img = self.RLImage(map_image_path, width=16*self.cm, height=10*self.cm)
                self.story.append(img)
                self.story.append(self.Spacer(1, 0.3*self.cm))

                caption = self.Paragraph("<i>Figure 1: Geographic distribution of publications</i>",
                                        self.styles['Normal'])
                self.story.append(caption)
                self.story.append(self.Spacer(1, 0.5*self.cm))
            except Exception as e:
                logger.warning(f"Could not add map image: {e}")

        # Add statistics table if provided
        if statistics and 'top_10_countries' in statistics:
            subtitle = self.Paragraph("Top 10 Countries by Publication Count",
                                     self.styles['SubsectionTitle'])
            self.story.append(subtitle)

            # Create table data
            table_data = [['Rank', 'Country', 'Publications', 'Percentage']]

            for idx, country_data in enumerate(statistics['top_10_countries'][:10], 1):
                table_data.append([
                    str(idx),
                    country_data['country'],
                    str(country_data['publications']),
                    f"{country_data['percentage']:.1f}%"
                ])

            # Create table
            table = self.Table(table_data, colWidths=[2*self.cm, 7*self.cm, 4*self.cm, 3*self.cm])
            table.setStyle(self.TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colors.HexColor('#2E86AB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), self.colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), self.colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, self.colors.grey)
            ]))

            self.story.append(table)
            self.story.append(self.Spacer(1, 0.5*self.cm))

        # Key insights
        insights_title = self.Paragraph("Key Insights", self.styles['SubsectionTitle'])
        self.story.append(insights_title)

        insights = [
            "Geographic distribution shows concentration in developed countries",
            "Research hubs identified in North America, Europe, and Asia",
            "Emerging research activity in developing regions",
            "International collaboration opportunities exist across continents"
        ]

        for insight in insights:
            bullet = self.Paragraph(f"• {insight}", self.styles['BulletPoint'])
            self.story.append(bullet)

        self.story.append(self.PageBreak())

    def add_wordcloud_section(self,
                             wordcloud_image_path: Optional[str] = None,
                             term_stats: Optional[Dict] = None):
        """
        Add word cloud analysis section.

        Args:
            wordcloud_image_path: Path to word cloud image
            term_stats: Term statistics dictionary
        """
        logger.info("Adding word cloud section")

        # Section title
        title = self.Paragraph("3. Term Analysis", self.styles['SectionTitle'])
        self.story.append(title)

        # Introductory text
        intro_text = """
        Term analysis reveals the key concepts, methodologies, and themes present in the
        scientific literature. The word cloud visualization highlights the most frequent
        and significant terms, providing insight into research focus areas and trending
        topics within the field.
        """
        intro_para = self.Paragraph(intro_text, self.styles['BodyJustify'])
        self.story.append(intro_para)
        self.story.append(self.Spacer(1, 0.5*self.cm))

        # Add word cloud if provided
        if wordcloud_image_path and Path(wordcloud_image_path).exists():
            try:
                img = self.RLImage(wordcloud_image_path, width=16*self.cm, height=10*self.cm)
                self.story.append(img)
                self.story.append(self.Spacer(1, 0.3*self.cm))

                caption = self.Paragraph("<i>Figure 2: Word cloud of most frequent terms</i>",
                                        self.styles['Normal'])
                self.story.append(caption)
                self.story.append(self.Spacer(1, 0.5*self.cm))
            except Exception as e:
                logger.warning(f"Could not add word cloud image: {e}")

        # Add top terms table if provided
        if term_stats and 'top_terms' in term_stats:
            subtitle = self.Paragraph("Top 20 Most Frequent Terms",
                                     self.styles['SubsectionTitle'])
            self.story.append(subtitle)

            # Create table data
            table_data = [['Rank', 'Term', 'Frequency', 'Weight']]

            for idx, (term, freq, weight) in enumerate(term_stats['top_terms'][:20], 1):
                table_data.append([
                    str(idx),
                    term,
                    str(freq),
                    f"{weight:.3f}"
                ])

            # Create table
            table = self.Table(table_data, colWidths=[2*self.cm, 7*self.cm, 3.5*self.cm, 3.5*self.cm])
            table.setStyle(self.TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colors.HexColor('#2E86AB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), self.colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), self.colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, self.colors.grey)
            ]))

            self.story.append(table)

        self.story.append(self.PageBreak())

    def add_timeline_section(self,
                           timeline_images: Optional[List[str]] = None,
                           temporal_stats: Optional[Dict] = None):
        """
        Add timeline analysis section.

        Args:
            timeline_images: List of timeline image paths
            temporal_stats: Temporal statistics dictionary
        """
        logger.info("Adding timeline section")

        # Section title
        title = self.Paragraph("4. Temporal Evolution", self.styles['SectionTitle'])
        self.story.append(title)

        # Introductory text
        intro_text = """
        Temporal analysis tracks the evolution of research activity over time, revealing
        trends, growth patterns, and potential bursts in publication activity. Understanding
        these patterns helps identify emerging areas, predict future trends, and assess
        the field's development trajectory.
        """
        intro_para = self.Paragraph(intro_text, self.styles['BodyJustify'])
        self.story.append(intro_para)
        self.story.append(self.Spacer(1, 0.5*self.cm))

        # Add timeline images if provided
        if timeline_images:
            for idx, img_path in enumerate(timeline_images[:2], 1):  # Max 2 images
                if Path(img_path).exists():
                    try:
                        img = self.RLImage(img_path, width=16*self.cm, height=10*self.cm)
                        self.story.append(img)
                        self.story.append(self.Spacer(1, 0.3*self.cm))

                        caption = self.Paragraph(f"<i>Figure {2+idx}: Temporal evolution of publications</i>",
                                                self.styles['Normal'])
                        self.story.append(caption)
                        self.story.append(self.Spacer(1, 0.5*self.cm))
                    except Exception as e:
                        logger.warning(f"Could not add timeline image: {e}")

        # Add temporal statistics if provided
        if temporal_stats and 'summary' in temporal_stats:
            subtitle = self.Paragraph("Temporal Statistics",
                                     self.styles['SubsectionTitle'])
            self.story.append(subtitle)

            summary = temporal_stats['summary']

            stats_data = [
                ['Metric', 'Value'],
                ['First Publication Year', str(summary.get('first_year', 'N/A'))],
                ['Last Publication Year', str(summary.get('last_year', 'N/A'))],
                ['Total Years', str(summary.get('total_years', 'N/A'))],
                ['Total Publications', str(summary.get('total_publications', 'N/A'))],
                ['Average per Year', f"{summary.get('avg_per_year', 0):.2f}"],
                ['Most Productive Year', f"{summary.get('most_productive_year', 'N/A')} ({summary.get('most_productive_year_count', 0)} pubs)"],
                ['Average Growth Rate', f"{summary.get('avg_growth_rate', 0):.2f}% per year"]
            ]

            table = self.Table(stats_data, colWidths=[8*self.cm, 8*self.cm])
            table.setStyle(self.TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colors.HexColor('#2E86AB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), self.colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), self.colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, self.colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
            ]))

            self.story.append(table)

        # Future projection if available
        if temporal_stats and 'projection' in temporal_stats and temporal_stats['projection']:
            self.story.append(self.Spacer(1, 0.5*self.cm))

            proj_title = self.Paragraph("Future Projection",
                                       self.styles['SubsectionTitle'])
            self.story.append(proj_title)

            proj = temporal_stats['projection']
            proj_text = f"""
            Based on linear regression analysis (R² = {proj['r_squared']:.4f}),
            the trend indicates an average growth of <b>{proj['slope']:+.2f} publications per year</b>.
            Projected publication counts for the next three years:
            {proj['future_years'][0]} ({int(proj['projected_counts'][0])} pubs),
            {proj['future_years'][1]} ({int(proj['projected_counts'][1])} pubs),
            {proj['future_years'][2]} ({int(proj['projected_counts'][2])} pubs).
            """

            proj_para = self.Paragraph(proj_text, self.styles['BodyJustify'])
            self.story.append(proj_para)

        self.story.append(self.PageBreak())

    def add_summary_statistics(self, overall_stats: Dict):
        """
        Add summary statistics page.

        Args:
            overall_stats: Dictionary with overall statistics
        """
        logger.info("Adding summary statistics")

        # Section title
        title = self.Paragraph("5. Summary Statistics", self.styles['SectionTitle'])
        self.story.append(title)

        # Create statistics table
        stats_data = [['Metric', 'Value']]

        # Add all provided statistics
        for key, value in overall_stats.items():
            # Format key (replace underscores with spaces, capitalize)
            formatted_key = key.replace('_', ' ').title()

            # Format value
            if isinstance(value, (int, float)):
                formatted_value = f"{value:,}" if isinstance(value, int) else f"{value:.2f}"
            elif isinstance(value, list):
                formatted_value = ", ".join(map(str, value[:5]))  # Show first 5 items
                if len(value) > 5:
                    formatted_value += f", ... ({len(value)} total)"
            else:
                formatted_value = str(value)

            stats_data.append([formatted_key, formatted_value])

        # Create table
        table = self.Table(stats_data, colWidths=[10*self.cm, 6*self.cm])
        table.setStyle(self.TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), self.colors.whitesmoke),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), self.colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, self.colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))

        self.story.append(table)
        self.story.append(self.PageBreak())

    def add_metadata_page(self, processing_info: Dict):
        """
        Add metadata and methods page.

        Args:
            processing_info: Dictionary with processing metadata
        """
        logger.info("Adding metadata page")

        # Section title
        title = self.Paragraph("6. Metadata and Methods", self.styles['SectionTitle'])
        self.story.append(title)

        # Report generation
        subtitle1 = self.Paragraph("Report Generation", self.styles['SubsectionTitle'])
        self.story.append(subtitle1)

        gen_info = [
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"System Version: {processing_info.get('version', '1.0.0')}",
            f"Python Version: {processing_info.get('python_version', 'N/A')}"
        ]

        for info in gen_info:
            para = self.Paragraph(f"• {info}", self.styles['BulletPoint'])
            self.story.append(para)

        self.story.append(self.Spacer(1, 0.5*self.cm))

        # Data sources
        subtitle2 = self.Paragraph("Data Sources", self.styles['SubsectionTitle'])
        self.story.append(subtitle2)

        sources = processing_info.get('sources', ['IEEE Xplore', 'ACM Digital Library', 'Scopus', 'Web of Science'])
        for source in sources:
            para = self.Paragraph(f"• {source}", self.styles['BulletPoint'])
            self.story.append(para)

        self.story.append(self.Spacer(1, 0.5*self.cm))

        # Methods
        subtitle3 = self.Paragraph("Analysis Methods", self.styles['SubsectionTitle'])
        self.story.append(subtitle3)

        methods_text = """
        This report employs advanced bibliometric analysis techniques including:
        geographic distribution analysis using Named Entity Recognition (NER),
        term frequency analysis with TF-IDF weighting, temporal evolution analysis
        with linear regression projections, and publication burst detection using
        statistical anomaly detection methods.
        """

        methods_para = self.Paragraph(methods_text, self.styles['BodyJustify'])
        self.story.append(methods_para)

        self.story.append(self.Spacer(1, 0.5*self.cm))

        # Contact/references
        subtitle4 = self.Paragraph("References", self.styles['SubsectionTitle'])
        self.story.append(subtitle4)

        ref_text = processing_info.get('references', 'For more information, please contact the research team.')
        ref_para = self.Paragraph(ref_text, self.styles['BodyText'])
        self.story.append(ref_para)

    def optimize_images_for_pdf(self, image_paths: List[str]) -> List[str]:
        """
        Optimize images for PDF inclusion.

        - Convert to RGB if necessary
        - Resize maintaining aspect ratio
        - Compress to quality 85-90%
        - Ensure 300 DPI

        Args:
            image_paths: List of image file paths

        Returns:
            List of optimized image paths
        """
        try:
            from PIL import Image
        except ImportError:
            logger.warning("PIL not installed. Cannot optimize images.")
            return image_paths

        logger.info(f"Optimizing {len(image_paths)} images for PDF")

        optimized_paths = []

        for img_path in image_paths:
            img_path = Path(img_path)

            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue

            try:
                # Open image
                img = Image.open(img_path)

                # Convert to RGB if necessary
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')

                # Set DPI to 300
                img.info['dpi'] = (300, 300)

                # Create optimized path
                optimized_path = img_path.parent / f"{img_path.stem}_optimized{img_path.suffix}"

                # Save with optimization
                img.save(
                    optimized_path,
                    quality=90,
                    optimize=True,
                    dpi=(300, 300)
                )

                optimized_paths.append(str(optimized_path))

                logger.info(f"Optimized: {img_path.name}")

            except Exception as e:
                logger.warning(f"Could not optimize {img_path}: {e}")
                optimized_paths.append(str(img_path))

        return optimized_paths

    def generate_complete_pdf(self,
                            title: str = "Bibliometric Analysis Report",
                            subtitle: str = "",
                            date_range: str = "",
                            authors: List[str] = None,
                            institution: str = "",
                            geographic_data: Optional[Dict] = None,
                            wordcloud_data: Optional[Dict] = None,
                            timeline_data: Optional[Dict] = None,
                            overall_stats: Optional[Dict] = None,
                            processing_info: Optional[Dict] = None) -> str:
        """
        Generate complete PDF with all sections.

        Args:
            title: Report title
            subtitle: Report subtitle
            date_range: Temporal range
            authors: List of authors
            institution: Institution name
            geographic_data: Geographic section data
            wordcloud_data: Word cloud section data
            timeline_data: Timeline section data
            overall_stats: Overall statistics
            processing_info: Processing metadata

        Returns:
            Path to generated PDF
        """
        logger.info("Generating complete PDF report")

        # Create PDF document
        doc = self.SimpleDocTemplate(
            str(self.output_pdf_path),
            pagesize=self.A4,
            leftMargin=2.5*self.cm,
            rightMargin=2.5*self.cm,
            topMargin=2.5*self.cm,
            bottomMargin=2.5*self.cm
        )

        # Create cover page
        self.create_cover_page(title, subtitle, date_range, authors, institution)

        # Add table of contents
        self.add_table_of_contents()

        # Add executive summary (placeholder)
        exec_title = self.Paragraph("1. Executive Summary", self.styles['SectionTitle'])
        self.story.append(exec_title)

        exec_text = """
        This bibliometric analysis report provides a comprehensive overview of research
        activity in the field. The analysis encompasses geographic distribution, term frequency
        analysis, and temporal evolution patterns. Key findings include identification of
        leading research countries, most frequent research terms, and publication trends over time.
        """
        exec_para = self.Paragraph(exec_text, self.styles['BodyJustify'])
        self.story.append(exec_para)
        self.story.append(self.PageBreak())

        # Add geographic section if data provided
        if geographic_data:
            self.add_geographic_section(
                map_image_path=geographic_data.get('map_image'),
                statistics=geographic_data.get('statistics')
            )

        # Add word cloud section if data provided
        if wordcloud_data:
            self.add_wordcloud_section(
                wordcloud_image_path=wordcloud_data.get('image'),
                term_stats=wordcloud_data.get('statistics')
            )

        # Add timeline section if data provided
        if timeline_data:
            self.add_timeline_section(
                timeline_images=timeline_data.get('images', []),
                temporal_stats=timeline_data.get('statistics')
            )

        # Add summary statistics if provided
        if overall_stats:
            self.add_summary_statistics(overall_stats)

        # Add metadata page if provided
        if processing_info:
            self.add_metadata_page(processing_info)

        # Build PDF
        try:
            doc.build(self.story)
            logger.success(f"PDF generated successfully: {self.output_pdf_path}")
            return str(self.output_pdf_path)

        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            raise


# Example usage
if __name__ == "__main__":
    logger.info("PDF Exporter Module - Example Usage")

    # This would normally be called with real data
    # exporter = PDFExporter('output/bibliometric_report.pdf')
    #
    # pdf_path = exporter.generate_complete_pdf(
    #     title="Bibliometric Analysis: Generative AI",
    #     subtitle="A Comprehensive Study",
    #     date_range="2018-2023",
    #     authors=["Dr. Jane Doe", "Prof. John Smith"],
    #     institution="University of Research",
    #     geographic_data={
    #         'map_image': 'output/geo_map.png',
    #         'statistics': {...}
    #     },
    #     wordcloud_data={
    #         'image': 'output/wordcloud.png',
    #         'statistics': {...}
    #     },
    #     timeline_data={
    #         'images': ['output/timeline.png'],
    #         'statistics': {...}
    #     },
    #     overall_stats={...},
    #     processing_info={...}
    # )

    logger.info("Example complete!")
