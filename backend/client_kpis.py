class ClientKPI:
    def __init__(self, user_id):
        self.user_id = user_id

    def calculate_conversion_rate(self, total_visits, total_conversions):
        """Calculate the conversion rate as a percentage."""
        if total_visits == 0:
            return 0
        return (total_conversions / total_visits) * 100

    def calculate_average_order_value(self, total_revenue, total_orders):
        """Calculate the average order value."""
        if total_orders == 0:
            return 0
        return total_revenue / total_orders

    def calculate_customer_retention_rate(self, retained_customers, total_customers):
        """Calculate the customer retention rate as a percentage."""
        if total_customers == 0:
            return 0
        return (retained_customers / total_customers) * 100

    def calculate_net_promoter_score(self, promoters, detractors, total_responses):
        """Calculate the Net Promoter Score (NPS)."""
        if total_responses == 0:
            return 0
        return ((promoters - detractors) / total_responses) * 100

    def facade(self):
        """function that returns a dictionary of KPIs"""
        return {
            "conversion_rate": self.calculate_conversion_rate(),
            "average_order_value": self.calculate_average_order_value(),
            "customer_retention_rate": self.calculate_customer_retention_rate(),
            "net_promoter_score": self.calculate_net_promoter_score()
        }